import os
import json
from audiostretchy.stretch import stretch_audio
from pydub import AudioSegment

def time_stretch(tts_dir, utterance_data, output_dir, durations):
    print("Performing time stretching on tts to match original length...")
    os.makedirs(output_dir, exist_ok=True)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
    speaker_dict = {}
    tts_stretched = {}
    tts_current_time = {}
    #Create list of empty lists - one for each speaker
    # durations_stretched = [[] for i in range(len(set([utt['Speaker'] for utt in utterance_data])))]
    durations_stretched = {}
    temp_tts_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_tts_dir, exist_ok=True)


    for utterance in utterance_data:
        speaker = utterance["Speaker"]
        duration = utterance["Duration"]
        speaker_index = alphabet.index(speaker)
        if speaker_index not in speaker_dict:
            tts_stretched[speaker_index] = AudioSegment.silent(duration=0)
            speaker_dict[speaker_index] = 0
            tts_current_time[speaker_index] = 0
            durations_stretched[speaker_index] = []
        

        tts_path = os.path.join(tts_dir, f"speaker{speaker_index}.wav")
    
        tts_start = tts_current_time[speaker_index]
        tts_duration = durations[speaker_index][speaker_dict[speaker_index]]
        tts_audio = AudioSegment.from_file(tts_path)[tts_start:tts_start + tts_duration]


        original_length = duration
        ratio = original_length / tts_duration

        speaker_dict[speaker_index] += 1

        output_path = os.path.join(output_dir, f"speaker{speaker_index}.wav")
        temp_tts_path = os.path.join(temp_tts_dir, f"temp_speaker_{speaker_index}_{speaker_dict[speaker_index]-1}.wav")

        tts_audio.export(temp_tts_path, format="wav")
        stretch_audio(temp_tts_path, output_path, ratio=ratio)
        # Trim silence of the end of the WAV file, that audiostretchy for some reason makes
        stretched = AudioSegment.from_file(output_path)
        trimmed = stretched[:original_length]
        trimmed.export(output_path, format="wav")
        trimmed.export(temp_tts_path, format="wav")

        durations_stretched[speaker_index].append(trimmed.duration_seconds * 1000)
        tts_current_time[speaker_index] += tts_duration

        tts_stretched[speaker_index] += AudioSegment.from_file(output_path)
    for speaker, audio in list(tts_stretched.items()):
        output_path = os.path.join(output_dir, f"speaker{speaker}.wav")
        stretched = audio
        stretched.export(output_path, format="wav")
        # print(f"Durations stretched speaker{speaker}: {durations_stretched[speaker]}")

if __name__ == "__main__":
    print("Testing time stretching...")
    video_output_dir = os.path.join("data", "processed", "video_3")
    original_dir = os.path.join(video_output_dir, "speaker_audio")
    tts_dir = os.path.join(video_output_dir, "speaker_tts")
    output_dir = os.path.join(video_output_dir, "tts_stretched")
    with open(os.path.join(video_output_dir, "transcription_video_3.json"), "r", encoding="utf-8") as f:
        utterance_data = json.load(f)
    durations = [[720, 4536, 1080, 2040, 3144, 2424, 3048, 2640, 2112, 5424, 4296, 1728, 1896], [624, 3768, 1824, 1080, 4152, 16224, 3072, 1776, 2592, 5952, 744, 4008]]
    time_stretch(tts_dir, utterance_data, output_dir, durations)