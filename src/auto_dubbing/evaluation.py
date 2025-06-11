import json
from collections import defaultdict
from itertools import permutations
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from jiwer import wer


# Load segments from diarization-style JSON
def load_segments(filepath, label_key="speaker", start_key="start", end_key="end"):
    with open(filepath, "r") as f:
        data = json.load(f)

    ann = Annotation()
    for entry in data:
        start = float(entry[start_key])
        end = float(entry[end_key])
        label = entry[label_key]
        ann[Segment(start, end)] = label
    return ann


def compute_der(reference_path, hypothesis_path, metric=None):
    reference = load_segments(reference_path, label_key="speaker")
    hypothesis = load_segments(hypothesis_path, label_key="speaker")

    der = metric(reference, hypothesis)
    details = metric.compute_components(reference, hypothesis)
    total = details["total"]

    return {
        "der": der,
        "confusion": details["confusion"] / total,
        "missed_detection": details["missed detection"] / total,
        "false_alarm": details["false alarm"] / total
    }, metric


# Generate cpWER-compatible JSON
def generate_cpwer_format(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    speaker_texts = defaultdict(list)
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "speaker" not in entry or "text" not in entry:
            continue
        speaker = entry["speaker"]
        text = entry["text"].strip()
        if text:
            speaker_texts[speaker].append(text)

    output = [{"speaker": spk, "text": " ".join(txts)} for spk, txts in speaker_texts.items()]
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


# Load cpWER JSON
def load_cpwer_json(path):
    with open(path) as f:
        data = json.load(f)
    return {entry["speaker"]: entry["text"] for entry in data}


def compute_cpwer(reference_path, hypothesis_path):
    ref = load_cpwer_json(reference_path)
    hyp = load_cpwer_json(hypothesis_path)

    ref_speakers = list(ref.keys())
    hyp_speakers = list(hyp.keys())

    if len(ref_speakers) != len(hyp_speakers):
        raise ValueError("Mismatch in number of speakers.")

    best_cpwer = float("inf")
    best_mapping = None

    for perm in permutations(hyp_speakers):
        total_words = 0
        total_errors = 0
        for ref_spk, hyp_spk in zip(ref_speakers, perm):
            r = ref[ref_spk]
            h = hyp[hyp_spk]
            total_words += len(r.split())
            total_errors += wer(r, h) * len(r.split())

        cpwer = total_errors / total_words if total_words else 1.0
        if cpwer < best_cpwer:
            best_cpwer = cpwer
            best_mapping = dict(zip(ref_speakers, perm))

    return {
        "cpwer": best_cpwer,
        "mapping": best_mapping
    }
