import os
import json
from glob import glob

def merge_transcripts(base_dir="data/processed", max_gap=1.0):
    video_dirs = sorted(glob(os.path.join(base_dir, "video_*")))

    for video_dir in video_dirs:
        input_path = os.path.join(video_dir, "transcript.json")
        output_path = os.path.join(video_dir, "transcript_con.json")

        if not os.path.exists(input_path):
            print(f"Skipping {video_dir}: transcript.json not found.")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        if not segments:
            print(f"No segments in {input_path}")
            continue

        merged_segments = []
        prev = segments[0]

        for curr in segments[1:]:
            time_gap = curr["start"] - prev["end"]
            if curr["speaker"] == prev["speaker"] and time_gap < max_gap:
                # Merge current segment into previous
                prev["end"] = curr["end"]
                prev["text"] = prev["text"].rstrip() + " " + curr["text"].lstrip()
                prev["translation"] = prev["translation"].rstrip() + " " + curr["translation"].lstrip()
            else:
                merged_segments.append(prev)
                prev = curr
        merged_segments.append(prev)  # Don't forget the last one

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_segments, f, indent=2, ensure_ascii=False)

        print(f"Merged transcript written to {output_path}")

merge_transcripts(max_gap=1.0)
