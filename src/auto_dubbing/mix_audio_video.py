import subprocess

def mix_audio_with_video(video_path, audio_path, output_path):
    # FFmpeg command to replace the audio
    command = [
        "ffmpeg",
        "-y",                   # Overwrite output without asking
        "-i", video_path,       # Input video
        "-i", audio_path,       # Input audio
        "-c:v", "copy",         # Copy video codec (no re-encoding)
        "-c:a", "aac",          # Encode audio with AAC codec
        "-map", "0:v:0",        # Use video from the first input
        "-map", "1:a:0",        # Use audio from the second input
        "-shortest",            # Match the duration of the shorter input
        output_path             # Output file
    ]

    # Run the command
    subprocess.run(command, check=True)

    print(f"Video with new audio saved at: {output_path}")