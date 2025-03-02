import os
import json
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Import your pipeline functions
from auto_dubbing.audio_extraction import extract_audio
from auto_dubbing.vocal_separation import separate_vocals
from auto_dubbing.vocal_processing import process_vocals
from auto_dubbing.transcribe import transcribe
from auto_dubbing.speaker_diarization import run_speaker_diarization
from auto_dubbing.translation import translate
from auto_dubbing.alignment import align_transcript_with_diarization, save_alignment_to_json

def main():
    load_dotenv()
    
    # Load configuration from config.yaml
    config = OmegaConf.load("config.yaml")
    
    # Define key paths and base name for the video
    video_path = config.paths.input_video
    processed_folder = config.paths.processed_folder
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(processed_folder, video_base)
    
    # Ensure the output directory exists
    os.makedirs(video_output_dir, exist_ok=True)
    
    # STEP 1: Extract audio from video
    input_audio = extract_audio(video_path, processed_folder)
    
    # STEP 2: Vocal separation (if needed)
    vocals_path, background_path = separate_vocals(input_audio, processed_folder)
    processed_vocals_path = process_vocals(vocals_path, processed_folder)
    
    # STEP 3: Transcription
    transcription_result = transcribe(
        audio_file=input_audio,  # Use the extracted audio file
        model_name=config.models.whisper.model_name,
        processed_folder=processed_folder,
        video_base=video_base
    )
    
    # Load or create transcription segments...
    transcription_segments_path = os.path.join(video_output_dir, f"transcription_segments_{video_base}.json")
    if not os.path.exists(transcription_segments_path):
        with open(transcription_segments_path, "w", encoding="utf-8") as f:
            import json
            json.dump(transcription_result["segments"], f, ensure_ascii=False, indent=2)
        print(f"Transcription segments saved to: {transcription_segments_path}")
    
    # STEP 4: Speaker Diarization
    hf_token = os.getenv("HF_ACCESS_TOKEN")
    diarization_result = run_speaker_diarization(
        audio_file=input_audio,  # IMPORTANT: Use the audio file, not the video file!
        hf_token=hf_token,
        processed_folder=processed_folder,
        video_base=video_base
    )
    
    # STEP 5: Translation
    deepl_key = os.getenv("DEEPL_API_KEY")
    translated_text = translate(
        text=transcription_result["text"],
        source_lang=config.translation.source_language,
        target_lang=config.translation.target_language,
        auth_key=deepl_key,
        processed_folder=processed_folder,
        video_base=video_base
    )
    
    # STEP 6: Alignment
    aligned_segments = align_transcript_with_diarization(
        whisper_result=transcription_result,
        diarization=diarization_result,
        translated_text=translated_text
    )
    
    # Save alignment output
    alignment_output_path = os.path.join(video_output_dir, f"alignment_{video_base}.json")
    save_alignment_to_json(aligned_segments, alignment_output_path)
    
    print("Pipeline completed successfully.")
    print(f"Alignment output is available in: {alignment_output_path}")


if __name__ == "__main__":
    main()