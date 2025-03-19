import os
import subprocess
from pydub import AudioSegment

# Function to calculate audio duration in seconds
def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    return len(audio) / 1000.0  # Duration in seconds

# Function to determine length adjustment factor
def calculate_length_adjust(source_duration, reference_duration):
    ratio = reference_duration / source_duration
    # Apply bounds
    if ratio > 1.25:
        return 1.25
    elif ratio < 0.75:
        return 0.75
    return ratio

# Function to run SEED-VC for voice conversion
def run_seed_vc(source, target, output_dir, length_adjust):
    os.makedirs(output_dir, exist_ok=True)

    command = [
        ".venv\\Scripts\\python", "seed-vc-main/inference.py",
        "--source", source,
        "--target", target,
        "--output", output_dir,
        "--diffusion-steps", "25",
        "--length-adjust", str(length_adjust),
        "--inference-cfg-rate", "0.7"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        print(f"Voice conversion completed. Output directory: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during voice conversion for {source} and {target}:")
        print(e.stderr)
        raise e

# Paths and directories
temp_dir = os.path.abspath("tempfiles")  # Use absolute path for temp files
output_base_dir = os.path.abspath("vc_segments")  # Base directory for SEED-VC outputs
speaker_audio_dir = os.path.abspath("speaker_audio")  # Directory for speaker audio files

# Dynamically map speaker audio files
speaker_to_file = {}

# List all files in the speaker_audio directory
for file_name in os.listdir(speaker_audio_dir):
    if file_name.endswith(".wav"):  # Ensure it's a WAV file
        # Extract the speaker name from the file name
        speaker_name = file_name.replace(".wav", "").replace("_", " ").title()
        # Map the speaker to the file path
        speaker_to_file[speaker_name] = os.path.join(speaker_audio_dir, file_name)

# Iterate over all segments
converted_segments = []  # To store paths of converted audio segments
for i, segment in enumerate(regrouped_segments):
    speaker = segment["speaker"]
    segment = segment["text"]

    # Files for this segment
    source_file = os.path.join(temp_dir, f"translated_segment{i}.wav")
    reference_length_file = os.path.join(temp_dir, f"segment{i}.wav")  # Used to calculate length_adjust
    target_speaker_file = speaker_to_file[speaker]  # Speaker-specific reference file for conversion
    output_dir = os.path.join(output_base_dir, f"segment_{i}")  # Unique directory for each segment

    # Check if files exist
    if not os.path.exists(source_file) or not os.path.exists(reference_length_file) or not os.path.exists(target_speaker_file):
        print(f"Skipping segment {i}: Missing file(s).")
        continue

    # Calculate durations
    source_duration = get_audio_duration(source_file)
    reference_duration = get_audio_duration(reference_length_file)

    # Determine length adjustment
    length_adjust = calculate_length_adjust(source_duration, reference_duration)

    # Run voice conversion and get the output file
    converted_file = run_seed_vc(source_file, target_speaker_file, output_dir, length_adjust)
    converted_segments.append(converted_file)

# Summary
print(f"Voice conversion completed for {len(converted_segments)} segments.")