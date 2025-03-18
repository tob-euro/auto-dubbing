import json
import os
from dotenv import load_dotenv

from auto_dubbing.transcribe import transcribe
from auto_dubbing.speaker_diarization import run_speaker_diarization
from auto_dubbing.translation import translate


def align_transcript_with_diarization(whisper_result, diarization, source_lang, target_lang, auth_key):
    """
    Align Whisper transcription segments with pyannote diarization segments.

    Args:
        whisper_result (dict): The full Whisper result, containing "segments" as a list.
        diarization (pyannote.core.Annotation): The diarization result from pyannote.
        translated_text (str, optional): If you've already translated the entire
                                         transcription in one pass, supply it here.
                                         Otherwise you can translate segment by segment
                                         or handle translation after alignment.

    Returns:
        list of dict: A list of aligned segments with speaker, start/end times,
                      text, and optional translated_text.
    """
    # Step 1: Gather Whisper segments
    try:
        whisper_segments = whisper_result["segments"]  # list of dicts: {"start", "end", "text"}
    except TypeError:
        whisper_segments = whisper_result  # If it's just a list of segments

    # Step 2: Gather diarization segments
    diarization_segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        diarization_segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker
        })

    # Sort them by start time just in case
    diarization_segments.sort(key=lambda x: x["start"])
    whisper_segments.sort(key=lambda x: x["start"])

    aligned_segments = []

    w_idx = 0
    d_idx = 0
    j = 0
    for i in range(len(whisper_segments)):
        wseg = whisper_segments[i]
        text = wseg["text"]
        start = wseg["start"]
        end = wseg["end"]
        if j < len(diarization_segments):
            speaker = diarization_segments[j]["speaker"]
            time_ = diarization_segments[j]["end"]
        if end > time_:
            j += 1
            speaker = diarization_segments[j]["speaker"]
            time_ = diarization_segments[j]["end"]
        translated_segment = translate(text, source_lang, target_lang, auth_key)
        aligned_segments.append({
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": text,
            "text_translated": translated_segment
        })
        
    # while w_idx < len(whisper_segments) and d_idx < len(diarization_segments):
    #     wseg = whisper_segments[w_idx]
    #     dseg = diarization_segments[d_idx]

    #     w_start, w_end = wseg["start"], wseg["end"]
    #     d_start, d_end = dseg["start"], dseg["end"]

    #     overlap_start = max(w_start, d_start)
    #     overlap_end = min(w_end, d_end)

    #     if overlap_start < overlap_end:
    #         # There's an overlap we can capture
    #         merged_text = wseg["text"]
    #         # Optionally, if you have a single big translation, you can later
    #         # subdivide it by time or do segment-level translations. For now, we
    #         # store the original text and maybe the full translated text:
    #         # translated_segment = translate(merged_text, source_lang, target_lang, auth_key)
    #         aligned_segments.append({
    #             "speaker": dseg["speaker"],
    #             "start": overlap_start,
    #             "end": overlap_end,
    #             "text_original": merged_text,
    #             "text_translated": None # translated_segment  # or you can fill if you do segment-level translation
    #         })

    #     # Whichever segment ends first, advance
    #     if w_end < d_end:
    #         w_idx += 1
    #     else:
    #         d_idx += 1

    # If you want to do segment-level translation now, you can do it in a loop:
    # (This is optional, especially if you prefer doing a single pass translation
    #  and storing it separately.)
    #
    # for seg in aligned_segments:
    #     seg["text_translated"] = translate_text_chunk(seg["text_original"], ...)

    return aligned_segments


def save_alignment_to_json(aligned_segments, output_path):
    """
    Save the aligned segments to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned_segments, f, ensure_ascii=False, indent=2)
    print(f"Alignment info saved to {output_path}")

if __name__ == "__main__":
    # Test the alignment function
    # Load the Whisper result from a JSON file
    load_dotenv()
    hf_token = os.getenv("HF_ACCESS_TOKEN")
    whisper_result_path = "data/processed/video_1/transcription_segments_video_1.json"
    with open(whisper_result_path, "r", encoding="utf-8") as f:
        whisper_result = json.load(f)
    diarization_result = run_speaker_diarization("data/processed/video_1/processed_vocals_video_1.wav", hf_token, "data/processed", "video_1")
    aligned_segments = align_transcript_with_diarization(
        whisper_result=whisper_result,
        diarization=diarization_result,
        translated_text=None,
        source_lang=None,
        target_lang=None,
        auth_key=None
    )
    print(aligned_segments)
    
    