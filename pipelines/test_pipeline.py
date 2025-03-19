import os
import json
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Import your pipeline functions
from auto_dubbing.audio_extraction import extract_audio
from auto_dubbing.vocal_separation import separate_vocals
from auto_dubbing.vocal_processing import process_vocals
from auto_dubbing.speaker_diarization import run_speaker_diarization
from auto_dubbing.translation import translate
from auto_dubbing.audio_slicing import split_audio_by_speaker
from auto_dubbing.tts import synthesize_text
from auto_dubbing.voice_conversion import run_seed_vc

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
    output_dir_input_audio = os.path.join(video_output_dir, f"extracted_audio_{video_base}.wav")
    if not os.path.exists(output_dir_input_audio):
        input_audio = extract_audio(video_path, processed_folder)
    else:
        input_audio = output_dir_input_audio
    
    # STEP 2: Vocal separation (if needed)
    output_dir_vocals_processed = os.path.join(video_output_dir, f"processed_vocals_{video_base}.wav")
    output_dir_background = os.path.join(video_output_dir, f"no_vocals_{video_base}.wav")
    if not (os.path.exists(output_dir_vocals_processed) and os.path.exists(output_dir_background)):
        vocals_path, background_path = separate_vocals(input_audio, processed_folder)
        processed_vocals_path = process_vocals(vocals_path, processed_folder, video_base)
    else:
        processed_vocals_path = output_dir_vocals_processed
        background_path = output_dir_background
    
    output_dir_transcription = os.path.join(video_output_dir, "transcription_{video_base}.json")
    if not os.path.exists(output_dir_transcription):
        # STEP 3: Speaker Diarization and Speaker Identification
        assemblyai_key = os.getenv("ASSEMBLY_API_KEY")
        transcript = run_speaker_diarization(
            audio_file=processed_vocals_path,
            assemblyai_key=assemblyai_key,
            processed_folder=processed_folder,
            video_base=video_base
        )
        # STEP 4: Translation
        # Save transcription segments (with translated text) with speaker identification to JSON file
        deepl_key = os.getenv("DEEPL_API_KEY")
    
        utterance_data = []
        for utterance in transcript.utterances:
            translated_text = translate(
                text=utterance.text,
                source_lang=config.translation.source_language,
                target_lang=config.translation.target_language,
                auth_key=deepl_key,
                processed_folder=processed_folder,
                video_base=video_base)
            utterance_data.append({"Speaker": utterance.speaker, "Text": utterance.text, "Start": utterance.start, "End": utterance.end, "Confidence": utterance.confidence, "Duration": int(utterance.end-utterance.start), "Translated_text": translated_text})
        transcription_segments_path = os.path.join(video_output_dir, f"transcription_{video_base}.json")
        with open(transcription_segments_path, "w", encoding="utf-8") as f:
            json.dump(utterance_data, f, ensure_ascii=False, indent=2)
        print(f"Transcription segments saved to: {transcription_segments_path}")
    else:
        with open(output_dir_transcription, "r", encoding="utf-8") as f:
            utterance_data = json.load(f)
    
    # STEP 5: Split processed vocal file into seperate files for each speaker
    speaker_translated_text_dict = split_audio_by_speaker(utterance_data, processed_vocals_path, video_output_dir)

    # STEP 6: Generate translated speech using tts
    tts_dir = os.path.join(video_output_dir, "speaker_tts")
    if not os.path.exists(tts_dir):
        translated_audio_files = synthesize_text(speaker_translated_text_dict, video_output_dir)

    
    # STEP 7: Apply voice conversion on translated audio files
    converted_files = []
    if not os.path.exists(os.path.join(video_output_dir, "vc")):
        os.makedirs(os.path.join(video_output_dir, "vc"), exist_ok=True)

        for i in range(len(list(speaker_translated_text_dict.keys()))):
            source = os.path.join(tts_dir, f"speaker{i}.wav")
            target = os.path.join(video_output_dir, "speaker_audio", f"speaker{i}.wav")
            output_dir = os.path.join(video_output_dir, "vc", f"speaker{i}")
            converted_file = run_seed_vc(source, target, output_dir)
            converted_files.append(converted_file)           
    

    



if __name__ == "__main__":
    main()