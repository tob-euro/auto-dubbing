from pydub import AudioSegment
import os

def split_audio_by_speaker(utterance_data, original_audio_path, output_dir):
    """
    Splits audio into separate tracks for each speaker and saves them as separate files.

    Args:
        grouped_segments (list): List of dictionaries containing speaker, text, start, and end timestamps.
        original_audio_path (str): Path to the original audio file.
        output_dir (str): Directory to save the split audio files.

    Returns:
        None
    """
    print("Slicing audio by speaker...")
    # Load the processed vocal audio
    audio = AudioSegment.from_wav(original_audio_path)

    # Dictionary to store audio for each speaker
    speaker_audio = {}
    speaker_translated_text = {}

    # Iterate through utternaces and append audio to the respective speaker
    for utterance in utterance_data:
        speaker = utterance["Speaker"]
        start_time = utterance["Start"]
        end_time = utterance["End"]

        # Initialize silent audio for the speaker if not already in dictionary
        if speaker not in speaker_audio:
            speaker_audio[speaker] = AudioSegment.silent(duration=0)
            speaker_translated_text[speaker] = [] #
    
        # Slice the audio and append to the corresponding speaker
        speaker_audio[speaker] += audio[start_time:end_time]
        speaker_translated_text[speaker] += [utterance["Translated_text"]] #

    # Ensure the output directory exists
    output_dir = os.path.join(output_dir, "speaker_audio")
    os.makedirs(output_dir, exist_ok=True)

    # Save the audio files for each speaker
    for i, (speaker, speaker_track) in enumerate(speaker_audio.items()):
        output_path = os.path.join(output_dir, f"speaker{i}.wav")
        speaker_track.export(output_path, format="wav")
        print(f"Saved {speaker} audio to: {output_path}")
    
    return speaker_translated_text