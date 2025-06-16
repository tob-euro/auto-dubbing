import os
import re
import csv
from auto_dubbing.evaluation import compute_der
from pyannote.metrics.diarization import DiarizationErrorRate
from moviepy import VideoFileClip


processed_dir = "data/processed"
baseline_dir = "data/baseline"
ground_truth_dir = "data/ground_truth"

results = []
baseline_results = []

metric = DiarizationErrorRate(collar=0.25)
baseline_metric = DiarizationErrorRate(collar=0.25)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def evaluate(video_id, video_path, metric_obj, results_list):
    video_num_match = re.search(r'\d+', video_id)
    if not video_num_match:
        print(f"⚠️  Could not parse number from {video_id}, skipping.")
        return

    video_num = video_num_match.group()
    gt_file = os.path.join(ground_truth_dir, f"video_{video_num}", f"ground_truth_video_{video_num}.json")
    hyp_file = os.path.join(video_path, "transcript_con.json")

    # fallback for baseline hypothesis file name
    if not os.path.exists(hyp_file):
        baseline_hyp_file = os.path.join(video_path, f"baseline_video_{video_num}.json")
        if os.path.exists(baseline_hyp_file):
            hyp_file = baseline_hyp_file

    if not os.path.exists(gt_file) or not os.path.exists(hyp_file):
        print(f"❌ Missing files for {video_id}, skipping.")
        print(f"   Expected GT file: {gt_file}")
        print(f"   Expected Hyp file: {hyp_file}")
        return

    der_result, _ = compute_der(gt_file, hyp_file, metric=metric_obj)

    result_entry = {
        "video": video_id,
        "DER": der_result["der"],
        "Confusion": der_result["confusion"],
        "Missed": der_result["missed_detection"],
        "FalseAlarm": der_result["false_alarm"]
    }

    results_list.append(result_entry)



def print_table(results_list, label):
    print(f"\n📊 {label}")
    header_fmt = "{:<12} {:>8} {:>9} {:>9} {:>9}"
    row_fmt = "{:<12} {:>8} {:>9} {:>9} {:>9}"

    print(header_fmt.format("Video", "DER", "Conf", "Miss", "FA"))
    print("-" * 50)
    for r in results_list:
        print(row_fmt.format(
            r["video"],
            f"{r['DER']:.2%}",
            f"{r['Confusion']:.2%}",
            f"{r['Missed']:.2%}",
            f"{r['FalseAlarm']:.2%}"
        ))

    # Compute and print averages
    n = len(results_list)
    if n > 0:
        avg_conf = sum(r["Confusion"] for r in results_list) / n
        avg_miss = sum(r["Missed"] for r in results_list) / n
        avg_fa = sum(r["FalseAlarm"] for r in results_list) / n
        print("-" * 50)
        print(row_fmt.format(
            "Average",
            "",
            f"{avg_conf:.2%}",
            f"{avg_miss:.2%}",
            f"{avg_fa:.2%}"
        ))


print("🔍 Evaluating main model...\n")
for video_id in sorted(os.listdir(processed_dir), key=natural_sort_key):
    video_path = os.path.join(processed_dir, video_id)
    if os.path.isdir(video_path):
        evaluate(video_id, video_path, metric, results)

print("🔍 Evaluating baseline model...\n")
for i in range(1, 25):
    video_id = f"video_{i}"
    video_path = os.path.join(baseline_dir, video_id)
    if os.path.isdir(video_path):
        evaluate(video_id, video_path, baseline_metric, baseline_results)

# Print tables
print_table(results, "Main Model Results")
print_table(baseline_results, "Baseline Results")

# Global metrics
def print_global_metrics(metric_obj, label):
    global_der = abs(metric_obj)
    print(f"\n🌍 {label} Global DER:")
    print(f"→ Global DER: {global_der:.2%}")

print_global_metrics(metric, "Main")
print_global_metrics(baseline_metric, "Baseline")