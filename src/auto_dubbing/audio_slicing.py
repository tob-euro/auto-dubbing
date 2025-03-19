from pydub import AudioSegment
import os

def split_audio_by_speaker(grouped_segments, original_audio_path, output_dir):
    """
    Splits audio into separate tracks for each speaker and saves them as separate files.

    Args:
        grouped_segments (list): List of dictionaries containing speaker, text, start, and end timestamps.
        original_audio_path (str): Path to the original audio file.
        output_dir (str): Directory to save the split audio files.

    Returns:
        None
    """
    # Load the processed vocal audio
    audio = AudioSegment.from_wav(original_audio_path)

    # Dictionary to store audio for each speaker
    speaker_audio = {}

    # Iterate through utternaces and append audio to the respective speaker
    for utterance in grouped_segments:
        speaker = utterance["Speaker"]
        start_time = utterance["Start"]
        end_time = utterance["End"]

        # Initialize silent audio for the speaker if not already in dictionary
        if speaker not in speaker_audio:
            speaker_audio[speaker] = AudioSegment.silent(duration=0)
    
        # Slice the audio and append to the corresponding speaker
        speaker_audio[speaker] += audio[start_time:end_time]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the audio files for each speaker
    for i, speaker, speaker_track in enumerate(speaker_audio.items()):
        output_path = os.path.join(output_dir, f"speaker_{i+1}.wav")
        speaker_track.export(output_path, format="wav")
        print(f"Saved {speaker} audio to: {output_path}")
    
    speakers = len(list(speaker_audio.keys()))
    
    return speakers