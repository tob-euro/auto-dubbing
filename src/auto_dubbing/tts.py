import os
import json
import logging
import shutil
import subprocess
from io import BytesIO
from dotenv import load_dotenv
from google.cloud import texttospeech
from pydub import AudioSegment
from audiostretchy.stretch import stretch_audio
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def synthesize_utterance_audio(transcript_path: str, output_dir: str, language_code: str = "da-DK",voice_name: str = "da-DK-Neural2-D", gender: texttospeech.SsmlVoiceGender = texttospeech.SsmlVoiceGender.FEMALE):
    """
    Read a transcription JSON and synthesize each utterance to a WAV.

    For each utterance dict in the top-level JSON list (must contain
    "Speaker" and "Translated_text"), writes:
        [output_dir]/speaker_audio/speaker_{Speaker}/tts_raw/{Speaker}_utt_{XX}.wav

    Args:
        transcript_path: Path to transcription JSON.
        output_dir:      Base folder where speaker_{X}/tts_raw/ lives.
        language_code:   Language code for TTS (defaults to Danish).
        voice_name:      Specific Google voice name.
        gender:          SSML gender selection.

    Returns:
        A dict mapping each speaker label (e.g. "A") → list of generated WAV paths.
    """
    logger.info("Loading transcript from %s", transcript_path)

    # 1) Load JSON
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

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

    # 3) Synthesize per utterance
    speaker_counts: dict[str, int]        = {}
    speaker_outputs: dict[str, list[str]] = {}

    logger.info("Starting TTS synthesis for %d utterances", len(transcript))

    for utterance in transcript:
        speaker_id  = utterance["speaker"]
        text = utterance["translation"]

        speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        idx = speaker_counts[speaker_id]

        # Synthesize
        logger.debug("Synthesizing speaker %s utt %02d", speaker_id, idx)
        input_msg = texttospeech.SynthesisInput(text=text)
        response  = client.synthesize_speech(
            request={"input": input_msg, "voice": voice, "audio_config": audio_config}
        )
        audio_seg = AudioSegment.from_mp3(BytesIO(response.audio_content))

        # Write out
        speaker_dir = os.path.join(output_dir, "speaker_audio", f"speaker_{speaker_id}", "tts_raw")
        os.makedirs(speaker_dir, exist_ok=True)
        filename = f"{speaker_id}_utt_{idx:02d}.wav"
        out_path = os.path.join(speaker_dir, filename)
        audio_seg.export(out_path, format="wav")

        speaker_outputs.setdefault(speaker_id, []).append(out_path)

    logger.info("TTS synthesis complete for %d utterances", len(transcript))
    return speaker_outputs


def time_stretch_tts(base_dir: str, transcript_path: str):
    """
    Time-stretch your TTS utterances so they match the original utterance timings.

    Expects this layout before you call it:

        {base_dir}/transcript.json
        {base_dir}/speaker_audio/speaker_{spk}/tts_raw/{spk}_utt_{XX}.wav

    It will produce:

        {base_dir}/speaker_audio/speaker_{spk}/tts_stretched/{spk}_utt_{XX}_stretched.wav

    Args:
        base_dir:        Root of your “processed/video_X” folder.
        transcript_path: Full path to transcript.json, a list of dicts
                         with keys “Speaker”, “Start”, “End”.
        output_dir:      (unused) kept for signature consistency

    Returns:
        A dict mapping each speaker label (e.g. "A") → list of
        final durations (ms) of the stretched files.
    """
    logger.info("Time-stretching TTS utterances")

    # Load transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Group by speaker
    by_speaker: dict[str, list[dict]] = {}
    for utterance in transcript:
        by_speaker.setdefault(utterance["speaker"], []).append(utterance)

    results: dict[str, list[int]] = {}
    for speaker_id, utts in by_speaker.items():
        logger.info("Processing speaker %s (%d utterances)", speaker_id, len(utts))

        # Correct paths under speaker_audio/
        raw_dir       = os.path.join(base_dir, "speaker_audio", f"speaker_{speaker_id}", "tts_raw")
        stretched_dir = os.path.join(base_dir, "speaker_audio", f"speaker_{speaker_id}", "tts_stretched")
        os.makedirs(stretched_dir, exist_ok=True)

        durations: list[int] = []
        for idx, utterance in enumerate(utts, start=1):
            target_ms = int((utterance["end"] - utterance["start"]) * 1000)
            src_path  = os.path.join(raw_dir, f"{speaker_id}_utt_{idx:02d}.wav")
            out_path  = os.path.join(stretched_dir, f"{speaker_id}_utt_{idx:02d}_stretched.wav")

            # load original to compute ratio
            orig = AudioSegment.from_file(src_path)
            ratio = (target_ms / len(orig)) if len(orig) > 0 else 1.0

            # stretch + trim
            stretch_audio(src_path, out_path, ratio=ratio)
            stretched = AudioSegment.from_file(out_path)[:target_ms]
            stretched.export(out_path, format="wav")

            got_ms = len(stretched)
            durations.append(got_ms)

        results[speaker_id] = durations

    logger.info("Time-stretching complete for %d speakers", len(results))
    return results


def run_seed_vc(source: str, target: str, output_dir: str, diffusion_steps: int = 100):
    """
    Run Seed-VC on a single stretched TTS utterance.
    """
    python_executable = "/opt/miniconda3/envs/seed-vc/bin/python"
    inference_script  = os.path.join("external", "seed-vc", "inference.py")

    command = [
        python_executable, inference_script,
        "--source", source,
        "--target", target,
        "--output", output_dir,
        "--diffusion-steps", str(diffusion_steps),
        "--length-adjust",  "1.0",
        "--inference-cfg-rate", "0.7"
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def voice_conversion_for_all(base_dir: str):
    """
    For every speaker under base_dir/speaker_audio, run Seed-VC on each
    stretched TTS file and emit *_vc.wav into tts_vc/.

    Expects:
        {base_dir}/speaker_audio/
            speaker_A/
                speaker_A.wav        # target voice
                tts_stretched/
                    A_utt_01_stretched.wav
                    ...
    Produces:
                tts_vc/
                    A_utt_01_vc.wav
                    ...
    """
    speaker_root = os.path.join(base_dir, "speaker_audio")
    logger.info("Running voice conversion under %s", speaker_root)

    if not os.path.isdir(speaker_root):
        logger.error("Expected speaker_audio directory at %s not found", speaker_root)
        return

    for entry in os.listdir(speaker_root):
        if not entry.startswith("speaker_"):
            continue

        spk_dir = os.path.join(speaker_root, entry)
        spk = entry.split("_", 1)[1]

        # target sample for this speaker
        target = os.path.join(spk_dir, f"{entry}.wav")
        stretched_dir = os.path.join(spk_dir, "tts_stretched")
        vc_dir       = os.path.join(spk_dir, "tts_vc")

        if not os.path.isfile(target):
            logger.warning("No target sample for %s, skipping", entry)
            continue
        if not os.path.isdir(stretched_dir):
            logger.warning("No stretched dir for %s, skipping", entry)
            continue

        os.makedirs(vc_dir, exist_ok=True)

        # find all stretched files for this speaker
        files = sorted([
            f for f in os.listdir(stretched_dir)
            if f.startswith(f"{spk}_utt_") and f.endswith("_stretched.wav")
        ])

        if not files:
            logger.warning("No stretched files for %s, skipping", entry)
            continue

        for idx, fname in enumerate(files, start=1):
            src = os.path.join(stretched_dir, fname)
            count = f"{idx:02d}"
            logger.info("Converting %s (utterance %s)", fname, count)

            temp_out = os.path.join(vc_dir, f"temp_{count}")
            try:
                run_seed_vc(src, target, temp_out)
            except subprocess.CalledProcessError as e:
                logger.error("Seed-VC failed on %s: %s", fname, e)
                shutil.rmtree(temp_out, ignore_errors=True)
                continue

            # move resulting .wav into final vc_dir
            wavs = [f for f in os.listdir(temp_out) if f.endswith(".wav")]
            if not wavs:
                logger.error("No output .wav in %s", temp_out)
                shutil.rmtree(temp_out, ignore_errors=True)
                continue

            final_name = f"{spk}_utt_{count}_vc.wav"
            shutil.move(os.path.join(temp_out, wavs[0]), os.path.join(vc_dir, final_name))
            shutil.rmtree(temp_out, ignore_errors=True)

    logger.info("Voice conversion complete.")