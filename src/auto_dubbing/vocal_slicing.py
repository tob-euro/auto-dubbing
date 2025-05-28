import os
import json
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def split_audio_by_speaker(transcript_path: str, vocals_path: str, output_dir: str):
    """
    Split a single vocals track into per-speaker WAVs based on the transcript.

    Reads a JSON array of utterances at `transcript_path` (each with keys
    "speaker", "start", and "end") and slices `vocals_path` accordingly.

    Args:
        transcript_path: Path to the JSON file containing utterances.
        vocals_path:     Path to the vocals WAV.
        output_dir:      Directory under which to create speaker folders.

    Returns:
        A dict mapping speaker label (e.g. "A", "B") → full path of that speaker’s WAV.
    """
    logger.info("Splitting audio by speakers")

    # Load the transcript JSON and the vocal audio
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    vocals = AudioSegment.from_wav(vocals_path)

    # Prepare the speaker_audio root
    speaker_root = os.path.join(output_dir, "speaker_audio")
    os.makedirs(speaker_root, exist_ok=True)

    # Collect audio per speaker
    speaker_audio: dict[str, AudioSegment] = {}
    for utterance in transcript:
        speaker_id = utterance["speaker"]
        start = int(utterance["start"] * 1000)
        end = int(utterance["end"] * 1000)

        seg = vocals[start:end]
        speaker_audio.setdefault(speaker_id, AudioSegment.silent(0))
        speaker_audio[speaker_id] += seg


    # Export each speaker file
    out_paths: dict[str, str] = {}
    for speaker_id, audio_segment in speaker_audio.items():
        speaker_folder = os.path.join(speaker_root, f"speaker_{speaker_id}")
        os.makedirs(speaker_folder, exist_ok=True)
        out_path = os.path.join(speaker_folder, f"speaker_{speaker_id}.wav")
        audio_segment.export(out_path, format="wav")
        logger.info("  Wrote %s → %s", speaker_id, out_path)
        out_paths[speaker_id] = out_path

    logger.info("Completed splitting %d speakers", len(out_paths))
    return out_paths

