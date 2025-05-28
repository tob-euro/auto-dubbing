from dotenv import load_dotenv
import os
import json
import assemblyai as aai

def run_speaker_diarization(audio_file: str, assemblyai_key: str):
    print("Running speaker diarization and transcription on audio...")
    aai.settings.api_key = assemblyai_key
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = transcriber.transcribe(audio_file, config=config)
    
    return transcript

if __name__ == "__main__":
    load_dotenv()
    
    assemblyai_key = os.getenv("ASSEMBLY_API_KEY")
    audio_file = "data/processed/video_2/extracted_audio_video_2.wav"
    processed_folder = "data/processed"
    video_base = "video_2"
    video_output_dir = os.path.join(processed_folder, video_base)
    
    transcript = run_speaker_diarization(audio_file, assemblyai_key)

    utterance_data = []
    for utterance in transcript.utterances:
        utterance_data.append({"Speaker": utterance.speaker, "Text": utterance.text, "Start": utterance.start, "End": utterance.end, "Confidence": utterance.confidence})
    transcription_segments_path = os.path.join(video_output_dir, f"test_transcription_segments_{video_base}.json")
    with open(transcription_segments_path, "w", encoding="utf-8") as f:
        json.dump(utterance_data, f, ensure_ascii=False, indent=2)
    print(f"Transcription segments saved to: {transcription_segments_path}")