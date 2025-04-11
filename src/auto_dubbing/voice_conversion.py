import os
import subprocess
from pydub import AudioSegment


def run_seed_vc(source, target, output_path):
    os.makedirs(output_path, exist_ok=True)

    command = [
        r"C:\Users\willi\anaconda3\envs\seed-vc\python", "seed-vc/inference.py",
        "--source", source,
        "--target", target,
        "--output", output_path,
        "--diffusion-steps", "125",
        "--length-adjust", str(1.0),
        "--inference-cfg-rate", "0.7"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        print(f"Voice conversion completed. Output directory: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during voice conversion for {source} and {target}:")
        print(e.stderr)
        raise e


    
if __name__ == "__main__":
    print("Testing voice conversion...")
    video_output_dir = os.path.join("data", "processed", "video_6")
    target_dir = os.path.join(video_output_dir, "speaker_audio")
    source_dir = os.path.join(video_output_dir, "tts_stretched")
    vc_dir = os.path.join(video_output_dir, "vc")
    source = os.path.join(source_dir, f"speaker0.wav")
    target = os.path.join(target_dir, f"speaker0.wav")
    output_dir = os.path.join(vc_dir, f"speaker0")
    converted_file = run_seed_vc(source, target, output_dir)