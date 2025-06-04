import os
import json
import logging
from pathlib import Path
import time
import deepl
import stable_whisper
import assemblyai as aai


# Silence httpx and assemblyai logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("assemblyai").setLevel(logging.WARNING)
# Silence DeepL client
logging.getLogger("deepl").setLevel(logging.WARNING)

# Create logger
logger = logging.getLogger(__name__)


def transcribe(audio_path: Path, output_dir: Path, model_name: str = "large-v3") -> str:
    """
    Transcribe 'audio_path' using OpenAI Whisper and write output to 'whisper.json'.

    Args:
        audio_path (Path): Path to the audio file.
        output_dir (Path): Directory to save 'whisper.json'.
        model_name (str): Whisper model to use.

    Returns:
        str: Path to the output JSON file.
    """
    logger.info("Starting Whisper transcription on %s", audio_path)
    model = stable_whisper.load_model(model_name)
    result = model.transcribe(
        str(audio_path),
    )

    language_code = "en"

    transcription = [
        {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip()
        }
        for seg in result.segments
    ]

    output_path = output_dir / "whisper_ts.json"
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
        tuple[str, str | None]: 
            - Path to diarization JSON file.
            - Detected language code (or None).
    """
    logger.info("Starting AssemblyAI diarization on %s", audio_file)

    # Setup AssemblyAI diarization
    aai.settings.api_key = auth_key
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True, language_detection=True)
    diarization = transcriber.transcribe(audio_file, config=config)

    # Extract speaker labels and timestamps in seconds into a list of dictionaries
    utterance_data = []
    for utterance in diarization.utterances:
        utterance_data.append({
            "Speaker": utterance.speaker,
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

    aligned = []
    discarded = 0

    for w_index, w_seg in enumerate(whisper_segments):
        w_start, w_end = w_seg["start"], w_seg["end"]
        max_overlap = 0
        best_speaker = None
        best_a_seg = None

        for a_seg in diarization_segments:
            a_start, a_end = a_seg["Start"], a_seg["End"]
            overlap = max(0, min(w_end, a_end) - max(w_start, a_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = a_seg["Speaker"]
                best_a_seg = a_seg

        if best_speaker and best_a_seg:
            true_start = best_a_seg["Start"] if len(aligned) == 0 else w_start
            aligned.append({
                "start": true_start,
                "end": w_end,
                "speaker": best_speaker,
                "text": w_seg["text"]
            })
        else:
            discarded += 1

    transcript_path = Path(output_dir) / "transcript.json"
    with transcript_path.open("w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    logger.info("Final transcript saved to %s", transcript_path)
    logger.info("Discarded %d non-overlapping Whisper segments", discarded)

    # Clean up intermediate files
    #Path(whisper_path).unlink(missing_ok=True)
    #Path(diarization_path).unlink(missing_ok=True)

    return str(transcript_path)


def translate(transcript_path: str, source_language: str, target_language: str, auth_key: str) -> None:
    """
    Translate each utterance in a transcription JSON in-place using DeepL. Queries the DeepL API
    with exponential backoff in case of rate limits. Writes the updated JSON back to the same file.

    Args:
        transcript_path:    Path to transcription JSON (list of utterance dicts).
        source_language:    DeepL source language code (e.g. "EN").
        target_language:    DeepL target language code (e.g. "DK").
        auth_key:           DeepL API authentication key.
    """
    logger.info("Translating %s from %s → %s via DeepL", transcript_path, source_language, target_language)    
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

# main file to test transcribe function
def main():
    # Example usage
    vocals = Path("data/processed/video_22/separated_audio/vocals.wav")
    output_dir = Path("data/processed/video_22")
    model_name = "large-v3"

    # Transcribe the audio
    transcribe(vocals, output_dir, model_name)

if __name__ == "__main__":
    main()