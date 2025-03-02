import os
import subprocess

def extract_audio(input_video: str, processed_folder: str) -> str:
    """
    Extracts audio from a video file using ffmpeg and saves it to a dedicated folder
    based on the video filename within the processed folder.
    
    Parameters:
        input_video (str): Path to the input video file (e.g. mp4, mov).
        processed_folder (str): Root folder where processed outputs should be stored.
        
    Returns:
        str: Path to the extracted audio file (wav).
    """
    print(f"Starting audio extraction from {input_video}...")

    # Determine the video basename without extension (e.g., "video_8")
    video_basename = os.path.splitext(os.path.basename(input_video))[0]
    
    # Create a dedicated folder for this video inside processed_folder
    video_output_folder = os.path.join(processed_folder, video_basename)
    os.makedirs(video_output_folder, exist_ok=True)
    
    # Define a descriptive output filename, e.g., "extracted_audio_video_8.wav"
    output_audio_filename = f"extracted_audio_{video_basename}.wav"
    output_audio = os.path.join(video_output_folder, output_audio_filename)
    
    # Construct the ffmpeg command to extract audio
    command = [
        "ffmpeg",
        "-y",                     # Automatically overwrite existing files
        "-loglevel", "error",     # Show errors only
        "-i", input_video,        # Specify the input video file
        "-vn",                    # Disable processing of the video stream
        "-acodec", "pcm_s16le",   # Use PCM 16-bit little-endian codec for WAV
        "-ar", "44100",           # Set audio sampling rate to 44.1 kHz
        "-ac", "2",               # Set number of audio channels to 2 (stereo)
        output_audio              # Output file path for the extracted audio
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Audio extraction complete: {output_audio}")
    except subprocess.CalledProcessError as e:
        print("Error during audio extraction:", e)
        raise RuntimeError(f"Error extracting audio from {input_video}") from e

    return output_audio
