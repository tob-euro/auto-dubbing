import os
import subprocess
import shutil

def separate_vocals(input_audio: str, processed_folder: str) -> tuple[str, str]:
    """
    Separates the input audio into two stems—vocals and background—using Demucs in 2-stem mode.
    Instead of leaving outputs in nested subfolders, this function moves the resulting files into
    the video-specific folder (<processed_folder>/<video_base>/).

    Parameters:
        input_audio (str): Path to the source audio file.
        processed_folder (str): Root folder for processed outputs.

    Returns:
        tuple: A tuple containing:
            - (str) Path to the isolated vocals track.
            - (str) Path to the background track (or None if not found).
    """
    print("Running Demucs for source separation...")

    # Determine base names.
    original_base = os.path.splitext(os.path.basename(input_audio))[0]
    adjusted_base = original_base.replace("extracted_audio_", "") if original_base.startswith("extracted_audio_") else original_base

    # We'll use the adjusted base as the intended video base for our output folder.
    video_base = adjusted_base

    # Create the output folder for this video (e.g., data/processed/video_8)
    output_dir = os.path.join(processed_folder, video_base)
    os.makedirs(output_dir, exist_ok=True)
    
    modeltype = "mdx_extra_q"

    # Run Demucs with --out set to the video folder.
    command = [
        "demucs",
        "-n", modeltype,       # Use old model (not hybrid transformer)
        "--two-stems", "vocals",  # Use two-stem mode (vocals vs. background)       
        "--out", output_dir,    # Write output to the video folder
        input_audio             # Input audio file
    ]
    subprocess.run(command, check=True)

    # Demucs creates a subfolder structure:
    #   <output_dir>/htdemucs/<folder_name>/vocals.wav
    # The folder name might be either the adjusted base or the original base.
    candidate1 = os.path.join(output_dir, modeltype, adjusted_base, "vocals.wav")
    candidate2 = os.path.join(output_dir, modeltype, original_base, "vocals.wav")

    if os.path.exists(candidate1):
        vocals_src = candidate1
    elif os.path.exists(candidate2):
        vocals_src = candidate2
    else:
        raise FileNotFoundError(f"Vocal file not found at expected locations: {candidate1} or {candidate2}")

    # Similarly for the background track.
    candidate1_bg = os.path.join(output_dir, modeltype, adjusted_base, "no_vocals.wav")
    candidate2_bg = os.path.join(output_dir, modeltype, original_base, "no_vocals.wav")
    if os.path.exists(candidate1_bg):
        background_src = candidate1_bg
    elif os.path.exists(candidate2_bg):
        background_src = candidate2_bg
    else:
        print("No separate background track found.")
        background_src = None

    # Define destination paths directly under the video folder.
    vocals_dest = os.path.join(output_dir, f"vocals_{video_base}.wav")
    background_dest = os.path.join(output_dir, f"no_vocals_{video_base}.wav")

    # Move the vocals file.
    shutil.move(vocals_src, vocals_dest)
    print(f"Moved vocal file to: {vocals_dest}")

    # Check and move background file if it exists.
    if background_src and os.path.exists(background_src):
        shutil.move(background_src, background_dest)
        print(f"Moved background file to: {background_dest}")
    else:
        background_dest = None

    # Remove the leftover Demucs folder (e.g., <output_dir>/htdemucs)
    htdemucs_dir = os.path.join(output_dir, modeltype)
    if os.path.exists(htdemucs_dir):
        shutil.rmtree(htdemucs_dir)
        print(f"Removed temporary Demucs folder: {htdemucs_dir}")

    return vocals_dest, background_dest

if __name__ == "__main__":
    processed_folder = os.path.join("data", "processed")
    input_audio = os.path.join(processed_folder, "video_8", "extracted_audio_video_8.wav")
    vocals_path, background_path = separate_vocals(input_audio=input_audio, processed_folder=processed_folder)