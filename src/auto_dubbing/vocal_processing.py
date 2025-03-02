from pydub import AudioSegment, effects
import numpy as np
import noisereduce as nr
import os

def process_vocals(vocals_path: str, processed_folder: str) -> str:
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
    filtered_vocals = normalized_vocals.high_pass_filter(80)

    # (Optional) mild low-pass filter if your audio has high-frequency noise
    # e.g., filtered_vocals = filtered_vocals.low_pass_filter(12000)

    # --- Step 3: Optional Noise Reduction ---
    print("Applying noise reduction using noisereduce...")
    # Convert pydub AudioSegment to raw samples
    samples = filtered_vocals.get_array_of_samples()
    sample_rate = filtered_vocals.frame_rate

    # Convert to float32 for noise reduction processing
    samples_np = np.array(samples).astype(np.float32)
    if filtered_vocals.channels == 2:
        # If stereo, reduce noise on each channel.
        samples_np = samples_np.reshape((-1, 2))
        # This is a simplistic approach. For real stereo, you might process channels separately.
        reduced_left = nr.reduce_noise(y=samples_np[:, 0], sr=sample_rate)
        reduced_right = nr.reduce_noise(y=samples_np[:, 1], sr=sample_rate)
        samples_np[:, 0] = reduced_left
        samples_np[:, 1] = reduced_right
        samples_np = samples_np.flatten()
    else:
        samples_np = nr.reduce_noise(y=samples_np, sr=sample_rate)

    # Convert back to int16 for pydub
    samples_int16 = samples_np.astype(np.int16)

    # Reconstruct a pydub AudioSegment
    filtered_vocals = AudioSegment(
        samples_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16 bits = 2 bytes
        channels=filtered_vocals.channels
    )

    # We derive the video base from the directory two levels up.
    video_base = os.path.basename(os.path.dirname(os.path.dirname(vocals_path)))
    output_dir = os.path.join(processed_folder, video_base, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"processed_vocals_{video_base}.wav"
    processed_vocals_path = os.path.join(output_dir, output_filename)

    filtered_vocals.export(processed_vocals_path, format="wav")

    print(f"Preprocessed vocals saved at: {processed_vocals_path}")
    return processed_vocals_path
