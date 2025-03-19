import os
import subprocess
from pydub import AudioSegment

# Function to run SEED-VC for voice conversion
def run_seed_vc(source, target, output_dir):
    source_duration = len(AudioSegment.from_wav(source)) / 1000.0
    target_duration = len(AudioSegment.from_wav(target)) / 1000.0
    ratio = target_duration / source_duration

    command = [
        ".venv\\Scripts\\python", "seed-vc-main/inference.py",
        "--source", source,
        "--target", target,
        "--output", output_dir,
        "--diffusion-steps", "25",
        "--length-adjust", str(ratio),
        "--inference-cfg-rate", "0.7"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        print(f"Voice conversion completed. Output directory: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error during voice conversion for {source} and {target}:")
        print(e.stderr)
        raise e