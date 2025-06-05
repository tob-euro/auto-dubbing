import os
import json
import logging
import shutil
import subprocess
from io import BytesIO
from google.cloud import texttospeech
from pydub import AudioSegment
from audiostretchy.stretch import stretch_audio

logger = logging.getLogger(__name__)

def trim_trailing_silence(audio: AudioSegment, silence_thresh: int = -40, chunk_size: int = 10) -> AudioSegment:
    """
    Trim silence from the end of the audio.

    Args:
        audio:         AudioSegment object.
        silence_thresh: Silence threshold in dBFS. Anything quieter is considered silence.
        chunk_size:    How many ms to scan at a time (higher = faster but less precise).

    Returns:
        Trimmed AudioSegment.
    """
    trim_ms = 0
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
    Read a transcription JSON and synthesize each utterance to a WAV.

    Args:
        transcript_path: Path to transcription JSON.
        output_dir:      Base folder where speaker_{X}/tts/ lives.
        language_code:   Language code for TTS (defaults to Danish).
        voice_name:      Specific Google voice name.
        gender:          SSML gender selection.

    Returns:
        A dict mapping each speaker label (e.g. "A") → list of generated WAV paths.
    """
    logger.info("Loading transcript from %s", transcript_path)

    # Load JSON
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Check if google credentials are set
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds or not os.path.isfile(creds):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS must point to your GCP JSON key")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds

    # Setup TTS client
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


def time_stretch_tts(base_dir: str, transcript_path: str):
    logger.info("Time-stretching TTS utterances")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    by_speaker: dict[str, list[dict]] = {}
    for utterance in transcript:
        by_speaker.setdefault(utterance["speaker"], []).append(utterance)

    for speaker_id, utts in by_speaker.items():
        logger.info("Processing speaker %s (%d utterances)", speaker_id, len(utts))

        tts_dir       = os.path.join(base_dir, "speaker_audio", f"speaker_{speaker_id}", "tts")
        stretched_dir = os.path.join(base_dir, "speaker_audio", f"speaker_{speaker_id}", "tts_stretched")
        os.makedirs(stretched_dir, exist_ok=True)

        MIN_RATIO = 0.5
        MAX_RATIO = 1.5

        for idx, utterance in enumerate(utts, start=1):
            target_ms = int((utterance["end"] - utterance["start"]) * 1000)
            src_path  = os.path.join(tts_dir, f"{speaker_id}_utt_{idx:02d}.wav")
            out_path  = os.path.join(stretched_dir, f"{speaker_id}_utt_{idx:02d}_stretched.wav")

            if not os.path.exists(src_path):
                logger.warning("Missing audio file: %s", src_path)
                continue

            orig = AudioSegment.from_file(src_path)
            if len(orig) == 0:
                logger.warning("Skipping empty audio file: %s", src_path)
                continue

            ratio = target_ms / len(orig)
            clamped_ratio = max(MIN_RATIO, min(MAX_RATIO, ratio))

            if clamped_ratio != ratio:
                logger.warning(
                    "Clamping stretch ratio for %s: %.2f -> %.2f (target=%dms, orig=%dms)",
                    src_path, ratio, clamped_ratio, target_ms, len(orig)
                )

            # Stretch audio using your stretch_audio function
            stretch_audio(src_path, out_path, ratio=clamped_ratio)

            if not os.path.exists(out_path):
                logger.error("Stretched audio not found: %s", out_path)
                continue

            stretched = AudioSegment.from_file(out_path)
            trimmed = stretched[:target_ms]

            # Warn if final duration is significantly short
            if len(trimmed) < 0.9 * target_ms:
                logger.warning(
                    "Trimmed audio is short (%dms < %dms) for %s",
                    len(trimmed), target_ms, out_path
                )

            trimmed.export(out_path, format="wav")

    logger.info("Time-stretching complete for %d speakers", len(by_speaker))



def split_audio_by_utterance(transcript_path: str, vocals_path: str, output_dir: str):
    """
    Split a vocals WAV into per-utterance WAVs based on speaker turns in transcript.

    Args:
        transcript_path: Path to the JSON file containing utterances.
        vocals_path:     Path to the vocals WAV.
        output_dir:      Directory where speaker_audio/speaker_{X}/utterances/ will be created.
    """
    logger.info("Splitting audio by utterances")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    vocals = AudioSegment.from_wav(vocals_path)

    speaker_root = os.path.join(output_dir, "speaker_audio")
    os.makedirs(speaker_root, exist_ok=True)

    speaker_counts: dict[str, int] = {}

    for utterance in transcript:
        speaker_id = utterance["speaker"]
        start_ms = int(utterance["start"] * 1000)
        end_ms = int(utterance["end"] * 1000)
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
    Build all reference audios for each utterance using a window of neighboring utterances.

    Args:
        base_dir: Root directory containing speaker_audio folders.
        reference_window: Number of neighboring utterances before and after to include.
    """
    speaker_root = os.path.join(base_dir, "speaker_audio")
    logger.info("Building reference audio for all speakers in %s", speaker_root)

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

        files = sorted([
            f for f in os.listdir(utt_dir)
            if f.startswith(f"{spk}_utt_") and f.endswith(".wav")
        ])

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

            combined = AudioSegment.empty()
            for ref in ref_paths:
                combined += AudioSegment.from_wav(ref)

            ref_out_path = os.path.join(ref_dir, f"{utt_id}_ref.wav")
            combined.export(ref_out_path, format="wav")

    logger.info("Finished building reference audio.")


def voice_conversion(source: str, target: str, output_dir: str):
    python_executable = "/opt/miniconda3/envs/seed-vc/bin/python"
    inference_script  = "inference_v2.py"

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
        "--intelligibility-cfg-rate", "0.8",
        "--similarity-cfg-rate", "0.9",
        "--convert-style", str(False),
        "--top-p", "0.9",
        "--temperature", "1.0",
        "--repetition-penalty", "1.0"
    ]

    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), "../../external/seed-vc")
    )


def process_all_voice_conversions(base_dir: str):
    """
    Perform Seed-VC voice conversion using prebuilt reference audio.

    Expects the following directory structure:
        speaker_audio/speaker_{X}/tts_stretched/{X}_utt_{XX}_stretched.wav
        speaker_audio/speaker_{X}/references/{X}_utt_{XX}_ref.wav

    Outputs:
        speaker_audio/speaker_{X}/tts_vc/{X}_utt_{XX}_vc.wav
    """
    speaker_root = os.path.join(base_dir, "speaker_audio")
    logger.info("Running voice conversion using prebuilt references under %s", speaker_root)

    if not os.path.isdir(speaker_root):
        logger.error("Expected speaker_audio directory at %s not found", speaker_root)
        return

    for entry in os.listdir(speaker_root):
        if not entry.startswith("speaker_"):
            continue

        spk = entry.split("_", 1)[1]
        spk_dir = os.path.join(speaker_root, entry)

        stretched_dir = os.path.join(spk_dir, "tts_stretched")
        ref_dir       = os.path.join(spk_dir, "references")
        vc_dir        = os.path.join(spk_dir, "tts_vc")

        if not os.path.isdir(stretched_dir) or not os.path.isdir(ref_dir):
            logger.warning("Missing required directories for speaker %s, skipping", spk)
            continue
        os.makedirs(vc_dir, exist_ok=True)

        files = sorted([
            f for f in os.listdir(stretched_dir)
            if f.startswith(f"{spk}_utt_") and f.endswith("_stretched.wav")
        ])
        if not files:
            logger.warning("No stretched files for speaker %s", spk)
            continue

        for idx, fname in enumerate(files, start=1):
            utt_id = f"{spk}_utt_{idx:02d}"
            src    = os.path.join(stretched_dir, fname)
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
