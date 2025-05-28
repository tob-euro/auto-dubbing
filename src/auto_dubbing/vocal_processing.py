import os
import logging
from pydub import AudioSegment, effects

logger = logging.getLogger(__name__)

def process_vocals(vocals_path: str, output_dir: str) -> str:
    """
    Preprocess the isolated vocals by normalizing loudness and applying
    a high-pass filter and a low-pass filter.

    Args:
        vocals_path: Path to the input vocals WAV.
        output_dir:  Directory where the processed file will be written.

    Returns:
        Path to the processed vocals WAV.
    """
    logger.info("processing vocals from %s", vocals_path)

    # Load and normalize vocals
    vocals = AudioSegment.from_file(vocals_path)
    vocals = effects.normalize(vocals)

    # Apply filters
    vocals = vocals.high_pass_filter(70)
    vocals = vocals.low_pass_filter(16_000)

    # Export
    processed_vocals_path = os.path.join(output_dir, "processed_vocals.wav")
    vocals.export(processed_vocals_path, format="wav")

    logger.info("Processed vocals saved to %s", processed_vocals_path)
    return processed_vocals_path