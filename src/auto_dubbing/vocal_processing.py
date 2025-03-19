from pydub import AudioSegment, effects
import numpy as np
import noisereduce as nr
import os

def process_vocals(vocals_path: str, processed_folder: str, video_base: str) -> str:
    """
    Applies general-purpose preprocessing to the isolated vocals track to improve clarity,
    without removing the character of the voice. This function:
      1. Normalizes loudness.
      2. Applies a mild high-pass filter (to remove low-frequency rumble).
      3. Optionally runs a simple noise reduction if 'noisereduce' is installed.

    Returns a path to the processed vocals file stored in a dedicated folder.
    """
    print("Preprocessing vocals for clarity...")

    # Load the vocals file
    vocal_segment = AudioSegment.from_file(vocals_path)

    # --- Step 1: Loudness Normalization ---
    # This ensures consistent average volume.
    normalized_vocals = effects.normalize(vocal_segment)

    # --- Step 2: Gentle High-Pass Filter (e.g., around 80Hz) ---
    # This helps remove rumble and sub-bass noise without affecting typical speech frequencies.
    # pydub’s high_pass_filter is in Hz, so passing 80 filters out frequencies below ~80Hz.
    filtered_vocals = normalized_vocals.high_pass_filter(70)

    # (Optional) mild low-pass filter if your audio has high-frequency noise
    filtered_vocals = filtered_vocals.low_pass_filter(16000)

    # We derive the video base from the directory two levels up.
    output_dir = os.path.join(processed_folder, video_base)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"processed_vocals_{video_base}.wav"
    processed_vocals_path = os.path.join(output_dir, output_filename)

    filtered_vocals.export(processed_vocals_path, format="wav")

    print(f"Preprocessed vocals saved at: {processed_vocals_path}")
    return processed_vocals_path

if __name__ == "__main__":
    # Test correct output directory
    output_dir = process_vocals("data/processed/video_6/vocals_video_6.wav", "data/processed", "video_6")
    print(f"Output directory: {output_dir}")
