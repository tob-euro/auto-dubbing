import os
import subprocess
import shutil
import glob
import logging
import json
from pathlib import Path
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def extract_audio(input_video: str, output_dir: str) -> str:
    """
    Extracts audio from a video file using ffmpeg and writes it as a WAV.

    Args:
        input_video: Path to the source video (mp4, mov, etc.).
        output_dir:  Directory where the extracted WAV will be written.

    Returns:
        The full path to the extracted WAV file.
    """
    # Build output path
    output_audio = os.path.join(output_dir, "extracted_audio.wav")

    logger.info("Extracting audio from %s → %s", input_video, output_audio)

    # Build and run ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",                    # overwrite
        "-loglevel", "error",    # only show errors
        "-i", input_video,       # input file
        "-vn",                   # no video
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
        "-ar", "44100",          # 44.1 kHz sampling
        "-ac", "2",              # stereo
        output_audio
    ]
    subprocess.run(cmd, check=True)

    logger.info("Audio successfully extracted to %s", output_audio)
    return output_audio


def separate_vocals(input_audio: str, output_dir: str) -> tuple[str, str]:
    """
    Separate a waveform into vocals and background audio using Demucs (2-stem mode).

    Args:
        input_audio: Path to the source WAV file.
        output_dir:  Base directory for processed outputs.

    Returns:
        A 2-tuple of (vocals_path, background_path).
    """
    logger.info("Running Demucs model on %s", input_audio)
    model = "mdx_extra_q"

    # Build and run Demucs command
    cmd = [
        "demucs",
        "-n", model,
        "--two-stems=vocals",
        "--out", output_dir,
        input_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Locate the temporary model folder
    model_dir = os.path.join(output_dir, model)

    # Glob for the two stems and pick the first match
    vocals_src = glob.glob(os.path.join(model_dir, "**", "vocals.wav"),     recursive=True)[0]
    background_src = glob.glob(os.path.join(model_dir, "**", "no_vocals.wav"), recursive=True)[0]

    # Prepare the separated folder
    sep_dir = os.path.join(output_dir, "separated_audio")
    os.makedirs(sep_dir, exist_ok=True)
    vocals_path = os.path.join(sep_dir, "vocals.wav")
    background_path = os.path.join(sep_dir, "background.wav")

    # Move them out
    shutil.move(vocals_src, vocals_path)
    shutil.move(background_src, background_path)
    logger.info("Saved vocal audio to %s", vocals_path)
    logger.info("Saved background audio to %s", background_path)

    # Clean up temp model folder
    shutil.rmtree(model_dir, ignore_errors=True)

    return vocals_path, background_path


def process_vocals(vocals_path: str, output_dir: str) -> str:
    """
    Preprocesses isolated vocals for transcription: normalizes volume, applies band-pass filtering,
    resamples to 16 kHz mono, and exports the result.

    Args:
        vocals_path: Path to the isolated vocals WAV file (e.g., from Demucs or source separation).
        output_dir:  Directory where the processed output will be saved.

    Returns:
        Path to the processed vocals WAV file.
    """
    logger.info("Processing vocals from %s", vocals_path)

    vocals = AudioSegment.from_file(vocals_path)

    # Normalize to target dBFS
    target_dBFS = -16.0
    change_in_dBFS = target_dBFS - vocals.dBFS
    vocals = vocals.apply_gain(change_in_dBFS)

    # Apply filters
    vocals = vocals.high_pass_filter(80)
    vocals = vocals.low_pass_filter(16000)

    # Resample for Whisper
    vocals = vocals.set_frame_rate(16000).set_channels(1)

    # Export
    processed_vocals_path = os.path.join(output_dir, "processed_vocals.wav")
    vocals.export(processed_vocals_path, format="wav")

    logger.info("Processed vocals saved to %s", processed_vocals_path)
    return processed_vocals_path


def mix_background_audio(transcript_path: str, extracted_audio_path: str, demucs_background_path: str, output_dir: str) -> str:
    """
    Remixes background audio using the original audio and the demucs isolated background audio.

    Args:
        transcript_path:         Path to the Whisper transcript JSON file.
        extracted_audio_path:    Path to the full extracted original audio (e.g., from a video).
        demucs_background_path:  Path to the background-only audio (e.g., from Demucs separation).
        output_dir:              Directory to save the final mixed output WAV file.

    Returns:
        Path to the reconstructed background mix WAV file.
    """
    # Load transcript
    with open(transcript_path, "r") as f:
        transcript = json.load(f)

    # Load audios
    orig_audio = AudioSegment.from_file(extracted_audio_path)
    bg_audio = AudioSegment.from_file(demucs_background_path)
    duration_ms = len(orig_audio)

    result = AudioSegment.empty()
    last_end_ms = 0

    for entry in transcript:
        start_ms = int(entry["start"] * 1000)
        end_ms = int(entry["end"] * 1000)

        # Clamp to audio duration
        start_ms = max(0, min(start_ms, duration_ms))
        end_ms = max(0, min(end_ms, duration_ms))

        # Add non-speech segment from original audio
        if start_ms > last_end_ms:
            non_speech = orig_audio[last_end_ms:start_ms]
            fade = min(500, len(non_speech) // 4)
            non_speech = non_speech.fade_in(fade).fade_out(fade)
            result += non_speech

        # Add speech segment from background audio
        speech = bg_audio[start_ms:end_ms]
        fade = min(500, len(speech) // 4)
        speech = speech.fade_in(fade).fade_out(fade)
        result += speech

        last_end_ms = end_ms

    # Add tail (non-speech) from end of last speech to end of audio
    if last_end_ms < duration_ms:
        tail = orig_audio[last_end_ms:]
        fade = min(500, len(tail) // 4)
        tail = tail.fade_in(fade).fade_out(fade)
        result += tail

    # Save the output
    output_path = Path(output_dir) / "background_mix.wav"
    result.export(output_path, format="wav")
    return str(output_path)


def combine_audio(base_dir: str, background_audio_path: str, transcript_path: str) -> str:
    """
    Overlays voice-converted clips onto the background track and writes the final mix.

    Args:
        base_dir:               Base directory where final mix will be written.
        background_audio_path:  Path to your background WAV.
        transcript_path:        Full path to the transcript JSON.

    Returns:
        The full path to the final mixed WAV file.
    """
    logger.info("Starting audio combination…")

    # Load transcript
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Load background track
    final_audio = AudioSegment.from_file(background_audio_path)
    os.makedirs(base_dir, exist_ok=True)

    # Overlay each VC clip
    speaker_counts: dict[str, int] = {}
    for utt in transcript:
        speaker   = utt["speaker"]
        start = int(utt["start"] * 1000)
        duration = int((utt["end"] - utt["start"]) * 1000)

        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        idx = speaker_counts[speaker]

        vc_path = os.path.join(
            base_dir,
            "speaker_audio",
            f"speaker_{speaker}",
            "tts_vc",
            f"{speaker}_utt_{idx:02d}_vc.wav"
        )

        if not os.path.isfile(vc_path):
            logger.warning("Missing VC clip %s, skipping…", vc_path)
            continue

        clip = AudioSegment.from_file(vc_path)[:duration]
        final_audio = final_audio.overlay(clip, position=start)

    # Export final mix
    output_wav = os.path.join(base_dir, "final_mix.wav")
    final_audio.export(output_wav, format="wav")
    logger.info("Final mix → %s", output_wav)

    return output_wav


def mix_audio_with_video(video_path: str, audio_path: str, output_video_path: str) -> str:
    """
    Replaces the audio track in video with `audio_path` using ffmpeg.

    Args:
        video_path:         Path to the source video.
        audio_path:         Path to the WAV audio to overlay.
        output_video_path:  Where to write the final dubbed video.

    Returns:
        The full path to the dubbed video file.
    """
    logger.info(
        "Mixing audio %s into video %s → %s",
        os.path.basename(audio_path),
        os.path.basename(video_path),
        output_video_path,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Build and run ffmpeg command
    cmd = [
        "ffmpeg",
        "-hide_banner",          # don’t show the initial banner/config dump
        "-loglevel", "error",    # only show actual errors
        "-y",                    # overwrite output without asking
        "-i", video_path,        # input video
        "-i", audio_path,        # input audio
        "-c:v", "copy",          # copy video stream
        "-c:a", "aac",           # encode audio as AAC
        "-map", "0:v:0",         # select video from first input
        "-map", "1:a:0",         # select audio from second input
        "-shortest",             # finish when the shorter stream ends
        output_video_path
    ]
    subprocess.run(cmd, check=True)

    logger.info("Dubbed video saved to %s", output_video_path)
    return output_video_path
