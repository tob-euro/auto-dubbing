import os
import json
import logging
import shutil
import subprocess
import tempfile
from io import BytesIO
from google.cloud import texttospeech
from pydub import AudioSegment
from dotenv import load_dotenv
from audiostretchy.stretch import stretch_audio

logger = logging.getLogger(__name__)

def trim_vc_start(base_dir: str, frames_to_trim: int = 3, fps: int = 30):
    """
    Trim the first few frames (converted to milliseconds) from the start of each voice-converted (VC) utterance WAV file to remove potential noise.

    Args:
        base_dir (str): Path to the root directory containing 'speaker_audio/*/tts_vc' folders.
        frames_to_trim (int): Number of video frames to trim from the start of each audio file (default: 3).
        fps (int): Frame rate of the source video, used to calculate trim duration in milliseconds (default: 30).
    """
    # Calculate trim time in miliseconds
    trim_ms = int((frames_to_trim / fps) * 1000)
    logger.info("Trimming %dms (%d frames at %dfps) from VC utterance starts", trim_ms, frames_to_trim, fps)

    # Go through all speakers
    speaker_dir = os.path.join(base_dir, "speaker_audio")
    for speaker in os.listdir(speaker_dir):
        vc_dir = os.path.join(speaker_dir, speaker, "tts_vc")
        if not os.path.isdir(vc_dir):
            continue

        # Go through and trim alle VC audiofiles
        for fname in os.listdir(vc_dir):
            if not fname.endswith(".wav"):
                continue
            path = os.path.join(vc_dir, fname)
            audio = AudioSegment.from_file(path)
            trimmed = audio[trim_ms:]
            trimmed.export(path, format="wav")
            logger.debug("Trimmed start of VC clip: %s", path)

    logger.info("VC start trimming complete.")

def trim_trailing_silence(audio: AudioSegment, silence_thresh: int = -40, chunk_size: int = 10) -> AudioSegment:
    """
    Trim trailing silence from the end of an AudioSegment.

    Args:
        audio (AudioSegment):        The audio to process.
        silence_thresh (int):        Silence threshold in dBFS; anything quieter is considered silence.
        chunk_size (int):            Number of milliseconds to scan at a time (higher = faster, less precise).

    Returns:
        AudioSegment: The trimmed audio segment with trailing silence removed.
    """
    # Initialize trim counter
    trim_ms = 0
    
    # Scan audio from the end in chunks
    while trim_ms < len(audio):
        chunk = audio[-(trim_ms + chunk_size): -trim_ms if trim_ms > 0 else None]
        if chunk.dBFS > silence_thresh:
            break
        trim_ms += chunk_size

    if trim_ms == 0:
        return audio
    return audio[:-trim_ms]


def tts(transcript_path: str, output_dir: str, language_code: str = "da-DK",voice_name: str = "da-DK-Neural2-D", gender: texttospeech.SsmlVoiceGender = texttospeech.SsmlVoiceGender.FEMALE):
    """
    Read a transcription JSON and synthesize each utterance to a WAV file using Google Cloud Text-to-Speech.
    Each utterance must have a "translation" field. Output WAV files are organized by speaker in subfolders.

    Args:
    transcript_path (str): Path to the transcription JSON file (each utterance must have a "translation" field).
    output_dir (str):      Base folder where speaker_{X}/tts/ subfolders will be created.
    language_code (str):   Language code for TTS (defaults to Danish, "da-DK").
    voice_name (str):      Specific Google voice name.
    gender (texttospeech.SsmlVoiceGender): SSML gender selection.

    Returns:
        dict: A mapping from each speaker label (e.g. "A") to a list of generated WAV file paths.
    """
    logger.info("Loading transcript from %s", transcript_path)

    # Load JSON transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Check if google credentials are set
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not os.path.isfile(creds):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS must point to your GCP JSON key")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

    # Setup TTS client and voice
    client = texttospeech.TextToSpeechClient()
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name,
        ssml_gender=gender,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    # Synthesize per utterance
    speaker_counts: dict[str, int] = {}

    logger.info("Starting TTS synthesis for %d utterances", len(transcript))

    for utterance in transcript:
        speaker_id = utterance["speaker"]
        text = utterance["translation"]

        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        idx = speaker_counts[speaker_id]

        # Synthesize
        input_msg = texttospeech.SynthesisInput(text=text)
        response = client.synthesize_speech(
            request={"input": input_msg, "voice": voice, "audio_config": audio_config}
        )
        audio_seg = AudioSegment.from_mp3(BytesIO(response.audio_content))
        audio_seg = trim_trailing_silence(audio_seg, silence_thresh=-40)

        # Write out
        speaker_dir = os.path.join(output_dir, "speaker_audio", f"speaker_{speaker_id}", "tts")
        os.makedirs(speaker_dir, exist_ok=True)
        filename = f"{speaker_id}_utt_{idx:02d}.wav"
        out_path = os.path.join(speaker_dir, filename)
        audio_seg.export(out_path, format="wav")

    logger.info("TTS synthesis complete for %d utterances", len(transcript))


def convert_to_pcm16(input_path: str, output_path: str):
    """
    Convert an audio file to 16-bit PCM WAV format using PyDub and ffmpeg.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the converted 16-bit PCM WAV file.

    Returns:
        None
    """
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_sample_width(2)  # 16-bit
    audio.export(output_path, format="wav")


def time_stretch_vc(base_dir: str, transcript_path: str):
    """
    Time stretches each voice converted tts segment to match original duration.
    The stretch ratio is clamped between [0.75, 1.25].

    Args:
        base_dir (Path): Path to base directory of video (data/processed/video_x).
        transcript_path (str): Path to the JSON file containing transcript.
    """

    logger.info("Time-stretching VC utterances")

    # Load the transcript JSON file
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Group utterances by speaker
    by_speaker: dict[str, list[dict]] = {}
    for utterance in transcript:
        by_speaker.setdefault(utterance["speaker"], []).append(utterance)

    # Process each speaker separately
    for speaker_id, utts in by_speaker.items():
        logger.info("Processing speaker %s (%d utterances)", speaker_id, len(utts))

        # Define input and output directories for this speaker
        vc_dir        = os.path.join(base_dir, "speaker_audio", f"speaker_{speaker_id}", "tts_vc")
        stretched_dir = os.path.join(base_dir, "speaker_audio", f"speaker_{speaker_id}", "tts_vc_stretched")
        os.makedirs(stretched_dir, exist_ok=True)

        MIN_RATIO = 0.75
        MAX_RATIO = 1.25

        # Process each utterance for this speaker
        for idx, utterance in enumerate(utts, start=1):
            # Calculate the target duration in milliseconds
            target_ms = int((utterance["end"] - utterance["start"]) * 1000)
            src_path  = os.path.join(vc_dir, f"{speaker_id}_utt_{idx:02d}_vc.wav")
            out_path  = os.path.join(stretched_dir, f"{speaker_id}_utt_{idx:02d}_vc_stretched.wav")

            # Skip if the source file does not exist
            if not os.path.exists(src_path):
                logger.warning("Missing VC file: %s", src_path)
                continue
            
            # Load the original VC audio
            orig = AudioSegment.from_file(src_path)
            if len(orig) == 0:
                logger.warning("Skipping empty VC file: %s", src_path)
                continue

            # Calculate and clamp the stretch ratio
            ratio = target_ms / len(orig)
            clamped_ratio = max(MIN_RATIO, min(MAX_RATIO, ratio))

            # Log if the ratio was clamped
            if clamped_ratio != ratio:
                logger.warning(
                    "Clamping stretch ratio for %s: %.2f -> %.2f (target=%dms, orig=%dms)",
                    src_path, ratio, clamped_ratio, target_ms, len(orig)
                )

            # Convert to PCM16 WAV (required by stretch_audio)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            convert_to_pcm16(src_path, tmp_path)
            # Perform time-stretching and export the result
            stretch_audio(tmp_path, out_path, ratio=clamped_ratio)
            os.remove(tmp_path)
            
            # Check if output was created
            if not os.path.exists(out_path):
                logger.error("Stretched VC audio not found: %s", out_path)
                continue

            logger.info("Exported stretched VC audio: %s", out_path)

    logger.info("VC time-stretching complete.")


def split_audio_by_utterance(transcript_path: str, vocals_path: str, output_dir: str):
    """
    Split a vocals WAV file into per-utterance WAVs based on speaker turns and timestamps in the transcript.
    Each utterance in the transcript must have 'speaker', 'start', and 'end' fields.
    Output WAV files are organized by speaker in subfolders.

    Args:
        transcript_path (str): Path to the JSON file containing utterances (with 'speaker', 'start', and 'end' fields).
        vocals_path (str):     Path to the vocals WAV file.
        output_dir (str):      Directory where speaker_audio/speaker_{X}/utterances/ will be created.
    """
    logger.info("Splitting audio by utterances")

    # Load transcript and audio
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    vocals = AudioSegment.from_wav(vocals_path)

    # Prepare output directory
    speaker_root = os.path.join(output_dir, "speaker_audio")
    os.makedirs(speaker_root, exist_ok=True)

    # Initialize speaker utterance counters
    speaker_counts: dict[str, int] = {}

    # Iterate over utterances
    for i, utterance in enumerate(transcript):
        speaker_id = utterance["speaker"]
        start_ms = int(utterance["start"] * 1000)
        end_ms = int(utterance["end"] * 1000)

        if i < len(transcript)-1:
            next_utterance = transcript[i+1]
            diff = int(next_utterance["start"] * 1000) - end_ms
            end_ms += min(500, diff)
        else:
            diff = len(vocals) - end_ms
            end_ms += min(500, diff)
        
        seg = vocals[start_ms:end_ms]

        speaker_dir = os.path.join(speaker_root, f"speaker_{speaker_id}", "utterances")
        os.makedirs(speaker_dir, exist_ok=True)

        # Increment this speaker's utterance count
        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        utt_idx = speaker_counts[speaker_id]

        filename = f"{speaker_id}_utt_{utt_idx:02d}.wav"
        out_path = os.path.join(speaker_dir, filename)
        seg.export(out_path, format="wav")

    logger.info("Finished splitting %d utterances across speakers", len(transcript))


def build_all_reference_audios(base_dir: str, reference_window: int = 1):
    """
    Build reference audio files for each utterance by concatenating the utterance with its neighboring utterances within a specified window.
    Reference audio files are saved in a 'references' subfolder for each speaker.

    Args:
        base_dir (str): Root directory containing 'speaker_audio' folders.
        reference_window (int): Number of neighboring utterances before and after to include for each reference audio.
    """
    speaker_root = os.path.join(base_dir, "speaker_audio")
    logger.info("Building reference audio for all speakers in %s", speaker_root)

    # Go through all speaker folders
    for entry in os.listdir(speaker_root):
        if not entry.startswith("speaker_"):
            continue
    
        spk = entry.split("_", 1)[1]
        spk_dir = os.path.join(speaker_root, entry)
        utt_dir = os.path.join(spk_dir, "utterances")
        ref_dir = os.path.join(spk_dir, "references")

        if not os.path.isdir(utt_dir):
            logger.warning("Missing utterance dir for %s, skipping", entry)
            continue
        os.makedirs(ref_dir, exist_ok=True)

        # Find all utterance files for the speaker
        files = sorted([
            f for f in os.listdir(utt_dir)
            if f.startswith(f"{spk}_utt_") and f.endswith(".wav")
        ])

        # For each utterance, build a reference file
        for idx, fname in enumerate(files):
            utt_id = f"{spk}_utt_{idx+1:02d}"
            idx_int = idx + 1

            ref_paths = []
            for offset in range(-reference_window, reference_window + 1):
                neighbor_idx = idx_int + offset
                if neighbor_idx < 1:
                    continue
                neighbor_fname = f"{spk}_utt_{neighbor_idx:02d}.wav"
                neighbor_path = os.path.join(utt_dir, neighbor_fname)
                if os.path.exists(neighbor_path):
                    ref_paths.append(neighbor_path)

            if not ref_paths:
                logger.warning("No valid references for %s", utt_id)
                continue
            
            # Combine and save the reference file
            combined = AudioSegment.empty()
            for ref in ref_paths:
                combined += AudioSegment.from_wav(ref)

            ref_out_path = os.path.join(ref_dir, f"{utt_id}_ref.wav")
            combined.export(ref_out_path, format="wav")

    logger.info("Finished building reference audio.")


def voice_conversion(source: str, target: str, output_dir: str):
    """
    Perform voice conversion on a source audio file using seed-vc, making the source sound like the target speaker.
    Runs the seed-vc inference script as a subprocess and saves the converted audio to the specified output directory.

    Args:
        source (str): Path to the source audio file (e.g., TTS segment to convert).
        target (str): Path to the target audio file (original speaker segment).
        output_dir (str): Directory where the voice-converted audio will be saved.

    Raises:
        subprocess.CalledProcessError: If the seed-vc inference subprocess fails.
    """
    load_dotenv()
    python_executable = os.environ["SEED_VC_PYTHON_PATH"]
    inference_script  = "seed-vc/inference.py"

    # Convert all paths to absolute
    source = os.path.abspath(source)
    target = os.path.abspath(target)
    output_dir = os.path.abspath(output_dir)

    command = [
        python_executable, inference_script,
        "--source", source,
        "--target", target,
        "--output", output_dir,
        "--diffusion-steps", "50",
        "--length-adjust", "1.0",
        "--inference-cfg-rate", "0.7"
    ]

    # Run the command and handle errors
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    except subprocess.CalledProcessError as e:
        print("Voice conversion subprocess failed!")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise


def process_all_voice_conversions(base_dir: str):
    """
    Perform Seed-VC voice conversion for all speakers using TTS outputs and prebuilt reference audio.

    For each speaker, expects:
        speaker_audio/speaker_{X}/tts/{X}_utt_{XX}.wav         # TTS-generated utterances
        speaker_audio/speaker_{X}/references/{X}_utt_{XX}_ref.wav  # Reference audio for each utterance

    Outputs:
        speaker_audio/speaker_{X}/tts_vc/{X}_utt_{XX}_vc.wav   # Voice-converted utterances

    Args:
        base_dir (str): Root directory containing the 'speaker_audio' folders.
    """
    speaker_root = os.path.join(base_dir, "speaker_audio")
    logger.info("Running voice conversion using references under %s", speaker_root)

    # Go throgh all the speaker folders
    for entry in os.listdir(speaker_root):
        if not entry.startswith("speaker_"):
            continue

        # Define paths
        spk = entry.split("_", 1)[1]
        spk_dir = os.path.join(speaker_root, entry)

        tts_dir = os.path.join(spk_dir, "tts")
        ref_dir = os.path.join(spk_dir, "references")
        vc_dir  = os.path.join(spk_dir, "tts_vc")

        if not os.path.isdir(tts_dir) or not os.path.isdir(ref_dir):
            logger.warning("Missing required directories for speaker %s, skipping", spk)
            continue
        os.makedirs(vc_dir, exist_ok=True)

        # Find alle TTS-files for the speaker
        files = sorted([
            f for f in os.listdir(tts_dir)
            if f.startswith(f"{spk}_utt_") and f.endswith(".wav")
        ])
        if not files:
            logger.warning("No TTS files for speaker %s", spk)
            continue
        
        # For each TTS-file, do voice conversion
        for idx, fname in enumerate(files, start=1):
            utt_id = f"{spk}_utt_{idx:02d}"
            src    = os.path.join(tts_dir, fname)
            ref    = os.path.join(ref_dir, f"{utt_id}_ref.wav")

            if not os.path.isfile(ref):
                logger.warning("Missing reference for %s → skipping", utt_id)
                continue

            logger.info("Converting %s using reference %s", fname, os.path.basename(ref))
            temp_out = os.path.join(vc_dir, f"temp_{idx:02d}")
            try:
                voice_conversion(src, ref, temp_out)
            except subprocess.CalledProcessError as e:
                logger.error("Seed-VC failed on %s: %s", utt_id, e)
                shutil.rmtree(temp_out, ignore_errors=True)
                continue

            vc_outputs = [f for f in os.listdir(temp_out) if f.endswith(".wav")]
            if not vc_outputs:
                logger.error("No output .wav for %s", utt_id)
                shutil.rmtree(temp_out, ignore_errors=True)
                continue

            final_path = os.path.join(vc_dir, f"{utt_id}_vc.wav")
            shutil.move(os.path.join(temp_out, vc_outputs[0]), final_path)
            shutil.rmtree(temp_out, ignore_errors=True)

    logger.info("Voice conversion complete.")