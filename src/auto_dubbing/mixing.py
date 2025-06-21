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
    Extract audio from a video file using ffmpeg and save it as a WAV file.

    Args:
        input_video (str): Path to the source video file (e.g., mp4, mov).
        output_dir (str): Directory where the extracted WAV file will be saved.

    Returns:
        str: The full path to the extracted WAV file.
    """
     # Convert to Path objects for better handling
    input_video = Path(input_video)
    output_dir = Path(output_dir)
    # Build output path
    output_audio = os.path.join(output_dir, "extracted_audio.wav")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting audio from %s → %s", input_video, output_audio)

    # Build ffmpeg command
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
    # Run ffmpeg command
    subprocess.run(cmd, check=True)

    logger.info("Audio successfully extracted to %s", output_audio)
    return output_audio


def separate_vocals(input_audio: str, output_dir: str) -> tuple[str, str]:
    """
    Separate a WAV audio file into vocals and background audio using Demucs (2-stem mode).

    Args:
        input_audio (str): Path to the source WAV file.
        output_dir (str): Base directory where processed outputs will be saved.

    Returns:
        tuple[str, str]: A tuple containing the paths to the separated vocals and background audio files.
    """
    # Model choice
    model = "mdx_extra_q"
    logger.info("Running Demucs model %r on %s", model, input_audio)
    
    # Build Demucs command
    cmd = [
        "demucs",
        "-n", model,
        "--two-stems=vocals",
        "--out", output_dir,
        input_audio
    ]
    # Run Demucs command
    subprocess.run(cmd, check=True)

    # Locate the temporary model folder
    model_dir = os.path.join(output_dir, model)

    # Glob for the two stems and pick the first match
    vocals_src     = glob.glob(os.path.join(model_dir, "**", "vocals.wav"),     recursive=True)[0]
    background_src = glob.glob(os.path.join(model_dir, "**", "no_vocals.wav"), recursive=True)[0]

    # Prepare the separated folder
    sep_dir = os.path.join(output_dir, "separated_audio")
    os.makedirs(sep_dir, exist_ok=True)
    vocals_path     = os.path.join(sep_dir, "vocals.wav")
    background_path = os.path.join(sep_dir, "background.wav")

    # Move them out
    shutil.move(vocals_src,     vocals_path)
    shutil.move(background_src, background_path)
    logger.info("Saved vocal audio to %s", vocals_path)
    logger.info("Saved background audio to %s", background_path)

    # Clean up temp model folder
    shutil.rmtree(model_dir, ignore_errors=True)
    logger.debug("Removed temporary folder %s", model_dir)

    #Return the paths
    return vocals_path, background_path


def combine_audio(base_dir: str, background_audio_path: str, transcript_path: str) -> str:
    """
    Overlay all voice-converted (VC) utterance clips onto a background audio track according to their timestamps in the transcript,
    and write the final mixed audio to "final_mix.wav".

    Args:
        base_dir (str): Base directory for processed outputs and speaker audio folders.
        background_audio_path (str): Path to the background audio WAV file.
        transcript_path (str): Path to the transcript JSON file containing utterance timings and speaker labels.

    Returns:
        str: Path to the final combined audio file ("final_mix.wav").
    """
    logger.info("Starting audio combination…")

    # Check and load transcript
    if not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # Load the background audio as the base for the final mix
    final_audio = AudioSegment.from_file(background_audio_path)
    os.makedirs(base_dir, exist_ok=True)

    speaker_counts: dict[str, int] = {} # Track how many utterances per speaker
    for utt in transcript:
        speaker = utt["speaker"]
        start = int(utt["start"] * 1000) # Convert start time to milliseconds

        # Increment utterance count for this speaker
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        idx = speaker_counts[speaker]

        # Build the path to the corresponding VC (voice-converted) stretched clip
        vc_path = os.path.join(
            base_dir,
            "speaker_audio",
            f"speaker_{speaker}",
            "tts_vc_stretched",  # use stretched!
            f"{speaker}_utt_{idx:02d}_vc_stretched.wav"
        )
        
        # Skip if the VC clip is missing
        if not os.path.isfile(vc_path):
            logger.warning("Missing VC clip %s, skipping…", vc_path)
            continue
        
        # Load the VC clip
        clip = AudioSegment.from_file(vc_path)
        # Overlay the VC clip onto the background at the correct position
        final_audio = final_audio.overlay(clip, position=start)

    # Export the final mixed audio to a WAV file
    output_wav = os.path.join(base_dir, "final_mix.wav")
    final_audio.export(output_wav, format="wav")
    logger.info("Final mix → %s", output_wav)

    return output_wav



def mix_audio_with_video(video_path: str, audio_path: str, output_video_path: str) -> str:
    """
    Replace the audio track in a video file with a new audio file using ffmpeg.

    Args:
        video_path (str): Path to the source video file.
        audio_path (str): Path to the WAV audio file to use as the new audio track.
        output_video_path (str): Path where the final dubbed video will be saved.

    Returns:
        str: The full path to the dubbed video file.
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

