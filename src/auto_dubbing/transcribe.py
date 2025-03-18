import whisper
import os

def transcribe(audio_file: str, model_name: str = "large-v3-turbo",
               processed_folder: str = None, video_base: str = None) -> dict:
    """
    Transcribes the entire audio file in a single pass using Whisper.
    Larger models produce better accuracy but may be slower.
    
    Args:
        audio_file (str): Path to the audio file (e.g., wav).
        model_name (str): Which Whisper model to load.
        processed_folder (str, optional): Root folder for processed outputs.
        video_base (str, optional): Base name of the video for naming output file.
        
    Returns:
        dict: Whisper transcription result with text and segments.
    """
    print(f"Loading Whisper model '{model_name}'")
    model = whisper.load_model(model_name)
    
    print("Transcribing entire audio in one shot...")
    result = model.transcribe(audio_file, verbose=False, language=None)
    
    # Save transcription to a file if output path info is provided.
    if processed_folder and video_base:
        output_dir = os.path.join(processed_folder, video_base)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"transcription_{video_base}.txt"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        print(f"Transcription saved to: {output_path}")
    
    return result

if __name__ == "__main__":
    # Test the transcription function
    audio_file = os.path.join("data", "processed", "video_8", "processed_vocals_video_8.wav")
    processed_path = os.path.join("data", "processed")
    result = transcribe(audio_file, "large-v3-turbo", processed_path, "video_8")
    print("Transcription result:", result["text"])