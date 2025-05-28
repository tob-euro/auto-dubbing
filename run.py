import os
import sys
import logging
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

from auto_dubbing.mixing import separate_vocals, extract_audio, combine_audio, mix_audio_with_video
from auto_dubbing.transcription import transcribe, speaker_diarization, align_speaker_labels, translate, ASSEMBLYAI_TO_DEEPL
from auto_dubbing.vocal_slicing import split_audio_by_speaker
from auto_dubbing.tts import synthesize_utterance_audio, time_stretch_tts, voice_conversion_for_all
from auto_dubbing.vocal_processing import process_vocals


logger = logging.getLogger(__name__)
from rich.logging import RichHandler
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_time=True, show_path=False)]
)


def load_keys() -> Tuple[str, str]:
    """Load API keys from environment variables."""
    load_dotenv()
    try:
        assembly_key = os.environ["ASSEMBLY_API_KEY"]
        deepl_key = os.environ["DEEPL_API_KEY"]
    except KeyError as e:
        logger.error("Missing environment variable: %s", e)
        sys.exit(1)
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
    # # Load configuration and keys
    config = OmegaConf.load("config.yaml")
    assembly_key, deepl_key = load_keys()
    target_lang = config.translation.target_language

    # # Prepare paths
    video_path, base_dir, output_folder = prepare_paths(config)

    # 1) Extract & separate
    audio_file = extract_audio(video_path, base_dir)
    vocals, background = separate_vocals(audio_file, base_dir)
    # vocals = process_vocals(vocals, base_dir)

    # 2) Transcribe & translate
    whisper_path = transcribe(vocals, base_dir)
    diarization_path, src_lang = speaker_diarization(str(vocals), assembly_key, str(base_dir))
    transcript_path = align_speaker_labels(whisper_path, diarization_path, str(base_dir))

    translate(
        transcript_path,
        ASSEMBLYAI_TO_DEEPL.get(src_lang.lower(), "AUTO"),
        target_lang,
        deepl_key
    )

    # 3) Slice, TTS, stretch, voice conversion
    split_audio_by_speaker(transcript_path, vocals, base_dir)
    synthesize_utterance_audio(transcript_path, base_dir)
    time_stretch_tts(base_dir, transcript_path)
    voice_conversion_for_all(base_dir)

    # 4) Combine & mix with video
    final_audio = combine_audio(base_dir, background, transcript_path)
    output_video = output_folder / f"{video_path.stem}_dubbed.mp4"
    mix_audio_with_video(video_path, final_audio, output_video)

    logger.info("Auto dubbing complete! Final video at: %s", output_video)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
