import os
import json
import logging
import warnings
from pathlib import Path
import time
import deepl
import stable_whisper
import assemblyai as aai

import whisper

# Silence httpx and assemblyai logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("assemblyai").setLevel(logging.WARNING)
# Silence DeepL client
logging.getLogger("deepl").setLevel(logging.WARNING)

# Suppress specific warnings from stable-whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
os.environ["KMP_WARNINGS"] = "FALSE"

# Create logger
logger = logging.getLogger(__name__)


def extend_short_segments(segments, min_duration=0.5):
    """
    Extend segments with very low duration from whisper output.

    Args:
        segments (Whisper object): Result from transcribe containing segments.
        min_duration (float): Minimum duration a segment is allowed to have.

    Returns:
        Whisper object: New Whisper object with extended segments.
    """
    for seg in segments:
        dur = seg.end - seg.start
        if dur < min_duration:
            # Stretch to minimum, center on original midpoint
            mid = (seg.start + seg.end) / 2
            seg.start = max(0, mid - min_duration / 2)
            seg.end = mid + min_duration / 2
    return segments


def transcribe(audio_path: Path, output_dir: Path) -> str:
    """
    Transcribe 'audio_path' using Stable Whisper and write output to 'whisper.json'.

    Args:
        audio_path (Path): Path to the audio file.
        output_dir (Path): Directory to save 'whisper.json'.
        model_name (str): Whisper model to use.

    Returns:
        tuple[str, str]: Path to the output JSON file and detected language code.
    """
    logger.info("Starting Whisper transcription on %s", audio_path)

    # setup model
    model = stable_whisper.load_model("large-v3")

    # Run word-leveltranscription
    result = model.transcribe(
        str(audio_path),
        vad=True,
        vad_threshold=0.45,
        min_word_dur=0.3,
        nonspeech_error=0.25,
        regroup=False,
        verbose=None
    )

    # Refine word-level timestamps
    model.refine(
        str(audio_path),
        result,
        word_level=True,
        precision=0.15,
        verbose=None
    )

    # Convert to segment-level transcription
    (
        result
        .ignore_special_periods()
        .clamp_max()
        .split_by_punctuation([('.', ' '), '。', '?', '？'])
        .split_by_gap(1.0)
        .clamp_max()
    )

    # Extend short segments
    result.segments = extend_short_segments(result.segments, min_duration=0.5)
    # Save detected language
    language_code = result.language

    # Convert segments to a list of dictionaries
    transcription = [
        {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        }
        for seg in result.segments
    ]

    # Write JSON output to output directory
    output_path = output_dir / "whisper.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transcription, f, ensure_ascii=False, indent=2)

    logger.info("Whisper transcription saved to %s", output_path)
    logger.info("Detected language: %s", language_code)

    return str(output_path), language_code


def speaker_diarization(audio_file: str, auth_key: str, output_dir: str) -> tuple[str, str | None]:
    """
    Run AssemblyAI diarization and write result to 'diarization.json'.

    Args:
        audio_file (str): Path to the WAV audio file.
        auth_key (str): AssemblyAI API key.
        output_dir (str): Directory to save 'diarization.json'.

    Returns:
        str: Path to diarization JSON file.
    """
    logger.info("Starting AssemblyAI diarization on %s", audio_file)

    # Setup AssemblyAI diarization
    aai.settings.api_key = auth_key
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True, language_detection=True)
    diarization = transcriber.transcribe(audio_file, config=config)

    # Extract speaker labels, text and timestamps in seconds into a list of dictionaries
    utterance_data = []
    for utterance in diarization.utterances:
        utterance_data.append({
            "Speaker": utterance.speaker,
            "Text": utterance.text.strip(),
            "Start": round(utterance.start / 1000, 3),
            "End": round(utterance.end / 1000, 3),
        })

    # Save the speaker diarization JSON
    diarization_path = os.path.join(output_dir, "diarization.json")
    with open(diarization_path, "w", encoding="utf-8") as f:
        json.dump(utterance_data, f, ensure_ascii=False, indent=2)

    logger.info("Saved diarization to %s", diarization_path)

    return str(diarization_path)


def align_speaker_labels(whisper_path: str, diarization_path: str, output_dir: str) -> str:
    """
    Merge Whisper transcription with AssemblyAI speaker labels using timestamp overlap.
    Discards any Whisper segments that do not overlap with diarized speaker segments.

    Args:
        whisper_path (str): Path to whisper.json.
        diarization_path (str): Path to diarization.json.
        output_dir (str): Directory to save final 'transcript.json'.

    Returns:
        str: Path to the final transcript JSON file.
    """
    logger.info("Loading Whisper segments from %s", whisper_path)
    with open(whisper_path, "r", encoding="utf-8") as f:
        whisper_segments = json.load(f)

    logger.info("Loading diarization segments from %s", diarization_path)
    with open(diarization_path, "r", encoding="utf-8") as f:
        diarization_segments = json.load(f)

    # Initialise output files
    aligned = []
    discarded = 0

    # Iterate over all Whisper segments with their index
    for w_index, w_seg in enumerate(whisper_segments):
        # Get start and end time for the current Whisper segment
        w_start, w_end = w_seg["start"], w_seg["end"]
        max_overlap = 0
        best_speaker = None
        best_a_seg = None

         # Iterate over all diarization segments to find the one with the largest overlap
        for a_seg in diarization_segments:
            a_start, a_end = a_seg["Start"], a_seg["End"]
            overlap = max(0, min(w_end, a_end) - max(w_start, a_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = a_seg["Speaker"]
                best_a_seg = a_seg

        # If a matching speaker segment was found
        if best_speaker and best_a_seg:
            true_start = best_a_seg["Start"] if len(aligned) == 0 else w_start
            # Add the aligned segment to the result list
            aligned.append({
                "start": true_start,
                "end": w_end,
                "speaker": best_speaker,
                "text": w_seg["text"]
            })
        else:
            # If no overlap was found, count the segment as discarded
            discarded += 1

    # Save the final transcript
    transcript_path = Path(output_dir) / "transcript.json"
    with transcript_path.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    logger.info("Final transcript saved to %s", transcript_path)
    logger.info("Discarded %d non-overlapping Whisper segments", discarded)


def translate(transcript_path: str, source_language: str, target_language: str, auth_key: str) -> None:
    """
    Translate each utterance in a transcription JSON in-place using DeepL. Queries the DeepL API
    with exponential backoff in case of rate limits. Writes the updated JSON back to the same file.

    Args:
        transcript_path (str):    Path to transcription JSON (list of utterance dicts).
        source_language (str):    DeepL source language code (e.g. "EN").
        target_language (str):    DeepL target language code (e.g. "DK").
        auth_key (str):           DeepL API authentication key.
    """
    logger.info("Translating %s from %s → %s via DeepL", transcript_path, source_language, target_language)
    # Setup DeepL translator    
    translator = deepl.Translator(auth_key)
    
    # Load the transcript JSON
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    
    # Translate each utterance with exponential backoff
    for utterance in transcript:
        text = utterance.get("text", "")
        if not text:
            continue

        max_retries = 5
        delay = 1  # delay in seconds

        for attempt in range(max_retries):
            try:
                translation = translator.translate_text(text, source_lang=source_language, target_lang=target_language)
                utterance["translation"] = translation.text
                break  # successful translation, exit retry loop
            except deepl.exceptions.TooManyRequestsException as e:
                logger.warning("Rate limit hit: %s. Retrying in %s seconds...", e, delay)
                time.sleep(delay)
                delay *= 2  # exponential backoff
            except deepl.exceptions.DeepLException as e:
                logger.error("DeepL API error: %s", e)
                break  # exit retry loop on other API errors
        else:
            logger.error("Failed to translate text after %s attempts: %s", max_retries, text)
            utterance["translation"] = ""  # or handle as needed

    # Write the updated utterances back to the JSON file
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    
    logger.info("Updated transcription with translations at %s", transcript_path)
    

def merge_segments(base_dir, max_gap=0.45):
    """
    Merge segments close to each other with same speaker in a transcript from a JSON file.
    Writes output transcript to a new JSON file ("transcript_con.json") 

    Args:
        base_dir (Path):    Path to base directory of video (data/processed/video_x) containing 'transcript.json'..
        max_gap (float):    Maximum allowed gap (in seconds) between segments with the same speaker to be merged.
    
    Returns:
        str: Path to the final transcript JSON file.
    """

    # Define input- and output paths
    input_path = os.path.join(base_dir, "transcript.json")
    output_path = os.path.join(base_dir, "transcript_con.json")

    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Initialise merging
    merged_segments = []
    prev = segments[0]

    # Go through and merge segments
    for curr in segments[1:]:
        time_gap = curr["start"] - prev["end"]
        if curr["speaker"] == prev["speaker"] and time_gap < max_gap:
            # Merge current segment into previous
            prev["end"] = curr["end"]
            prev["text"] = prev["text"].rstrip() + " " + curr["text"].lstrip()
        else:
            merged_segments.append(prev)
            prev = curr
    merged_segments.append(prev)  # Don't forget the last one

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_segments, f, indent=2, ensure_ascii=False)

    logger.info(f"Merged transcript written to {output_path}")

    return str(output_path)

# main function to test speaker diarization
def main():
    audio_path = Path("data/processed/Video_19/separated_audio/vocals.wav")
    output_dir = Path("")
    auth_key = "6dfee901a65e46569c33c517749dd225"

    # Run speaker diarization
    speaker_diarization(str(audio_path), auth_key, str(output_dir))

# run main
if __name__ == "__main__":
    main()