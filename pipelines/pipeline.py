import os
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Import your pipeline functions
from auto_dubbing.audio_extraction import extract_audio
from auto_dubbing.vocal_separation import separate_vocals
from auto_dubbing.vocal_processing import process_vocals
from auto_dubbing.transcribe import transcribe
from auto_dubbing.speaker_diarization import run_speaker_diarization
from auto_dubbing.translation import translate
from auto_dubbing.alignment import align_transcript_with_diarization

def main():
    load_dotenv()
    
    # Load configuration
    config = OmegaConf.load("config.yaml")
    
    # Derive key paths and video base name
    video_path = config.paths.input_video
    processed_folder = config.paths.processed_folder
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    
    # Run the pipeline steps quietly
    input_audio = extract_audio(video_path, processed_folder)
    vocals_path, background_path = separate_vocals(input_audio, processed_folder)
    processed_vocals_path = process_vocals(vocals_path, processed_folder)
    transcription_result = transcribe(
        audio_file=input_audio,
        model_name=config.models.whisper.model_name,
        processed_folder=processed_folder,
        video_base=video_base
    )
    transcribed_text = transcription_result["text"]
    hf_token = os.getenv("HF_ACCESS_TOKEN")
    diarization_result = run_speaker_diarization(
        processed_vocals_path, hf_token, processed_folder, video_base
    )
    deepl_key = os.getenv("DEEPL_API_KEY")
    translated_text = translate(
        text=transcribed_text,
        source_lang=config.translation.source_language,
        target_lang=config.translation.target_language,
        auth_key=deepl_key,
        processed_folder=processed_folder,
        video_base=video_base
    )

    # align_transcript_with_diarization(transcription_result, diarization_result)
    
    # Minimal summary output to the terminal.
    output_path = os.path.join(processed_folder, video_base)
    print("Pipeline completed successfully.")
    print(f"Processed outputs are available in: {output_path}")

if __name__ == "__main__":
    main()
