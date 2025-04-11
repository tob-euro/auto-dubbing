from pydub import AudioSegment
import os
import glob
import json

def combine_audio(vc_dir, background_audio_path, utterance_data, output_dir):
    final_audio = AudioSegment.from_file(background_audio_path)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
    speaker_dict = {}
    for utterance in utterance_data:
        start = utterance["Start"]
        speaker = utterance["Speaker"]
        duration = utterance["Duration"]


        speaker_index = alphabet.index(speaker)
        if speaker_index not in speaker_dict:
            speaker_dict[speaker_index] = 0
        vc_speaker_path = os.path.join(vc_dir, f"speaker{speaker_index}", f"vc_speaker{speaker_index}_speaker{speaker_index}_1.0_125_0.7.wav")


        speaker_start = speaker_dict[speaker_index]
        speaker_end = speaker_start + duration

        vc_segment = AudioSegment.from_file(vc_speaker_path)
        final_audio = final_audio.overlay(vc_segment[speaker_start:speaker_end], position=start)

        speaker_dict[speaker_index] = speaker_end
    output_path = os.path.join(output_dir, "output_video.wav")
    final_audio.export(output_path, format="wav")
    print(f"Final output audio saved to:" + output_path)
    return final_audio

if __name__ == "__main__":
    video_base = os.path.join("data", "processed", "video_6")
    background_audio_path = os.path.join(video_base, "no_vocals_video_6.wav")
    with open(os.path.join(video_base, "transcription_video_6.json"), "r", encoding="utf-8") as f:
        utterance_data = json.load(f)
    output_dir = os.path.join("data", "output")
    vc_dir = os.path.join(video_base, "vc")
    fa = combine_audio(vc_dir, background_audio_path, utterance_data, output_dir)
    print(fa.duration_seconds)