from pyannote.audio import Pipeline
import os

def run_speaker_diarization(audio_file: str, hf_token: str,
                            processed_folder: str, video_base: str) -> any:
    """
    Runs speaker diarization on the given audio file using pyannote's
    speaker-diarization-3.1 pipeline. Saves the RTTM file directly under the video folder.
    
    Args:
        audio_file (str): Path to the audio file (vocals recommended).
        hf_token (str): Your Hugging Face token for pyannote models.
        processed_folder (str): Root folder for processed outputs.
        video_base (str): Base name of the video for naming the output file.
        
    Returns:
        The diarization annotation object.
    """
    print("Running pyannote speaker diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    diarization = diarization_pipeline(audio_file)
    
    # Save the RTTM file directly under the video folder.
    output_dir = os.path.join(processed_folder, video_base)
    os.makedirs(output_dir, exist_ok=True)
    rttm_filename = f"diarization_{video_base}.rttm"
    rttm_output = os.path.join(output_dir, rttm_filename)
    
    with open(rttm_output, "w") as f:
        diarization.write_rttm(f)
    
    print("Diarization complete. RTTM saved to:", rttm_output)
    return diarization