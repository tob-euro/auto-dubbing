import os
import sys
import logging
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from rich.logging import RichHandler

from auto_dubbing.mixing import extract_audio, separate_vocals, combine_audio, mix_audio_with_video
from auto_dubbing.transcription import transcribe, speaker_diarization, align_speaker_labels, translate, merge_segments
from auto_dubbing.tts import trim_vc_start, tts, time_stretch_vc, split_audio_by_utterance, process_all_voice_conversions, build_all_reference_audios

logger = logging.getLogger(__name__)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_time=True, show_path=False)]
)


def load_keys() -> Tuple[str, str]:
    """Load API keys from environment variables."""
    load_dotenv()
    assembly_key = os.environ["ASSEMBLY_API_KEY"]
    deepl_key = os.environ["DEEPL_API_KEY"]
    return assembly_key, deepl_key


def prepare_paths(config: DictConfig) -> Tuple[Path, Path, Path]:
    """Create and return video_path, base_dir, and output_folder."""
    video_path = Path(config.paths.input_video).expanduser()
    video_name = video_path.stem
    processed = Path(config.paths.processed_folder)
    output = Path(config.paths.output_folder)
    base_dir = processed / video_name

    for folder in (processed, output, base_dir):
        folder.mkdir(parents=True, exist_ok=True)

    logger.debug("Prepared folders: %s, %s, %s", processed, output, base_dir)
    return video_path, base_dir, output


def run_pipeline() -> None:
    """Full auto-dubbing pipeline: extraction → processing → mixing."""
    # Load configuration and keys
    config = OmegaConf.load("config.yaml")
    assembly_key, deepl_key = load_keys()

    # Prepare paths
    video_path, base_dir, output_folder = prepare_paths(config)

    # 1) Extract & separate audio
    audio = extract_audio(video_path, base_dir)
    vocals, background = separate_vocals(audio, str(base_dir))

    # 2) Transcribe & translate vocals
    transcript, language = transcribe(vocals, base_dir)
    diarization = speaker_diarization(str(vocals), assembly_key, base_dir)
    align_speaker_labels(transcript, diarization, base_dir)
    full_transcript = merge_segments(base_dir, max_gap=0.45)
    translate(full_transcript, language.upper(), config.translation.target_language, deepl_key)

    # 3) TTS, stretch, build references, voice conversion
    split_audio_by_utterance(full_transcript, vocals, base_dir)
    build_all_reference_audios(base_dir, reference_window=1)
    tts(full_transcript, base_dir)
    process_all_voice_conversions(base_dir)
    trim_vc_start(base_dir)
    time_stretch_vc(base_dir, full_transcript)

    # 4) Remix background, mix vocals & background, combine audio with video
    final_audio = combine_audio(base_dir, background, full_transcript)
    output_video = output_folder / f"{video_path.stem}_dubbed.mp4"
    mix_audio_with_video(video_path, final_audio, output_video)

    logger.info("Auto dubbing complete! Final video at: %s", output_video)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)