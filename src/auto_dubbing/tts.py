from google.cloud import texttospeech
from pydub import AudioSegment
from io import BytesIO
import os

def synthesize_text(speaker_text_dict, output_dir):
    """Synthesizes speech from the input string of text."""
    print("Synthesizing tts voices from translated transcript...")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dtumlops-448114-5aa70f0364ab.json"
    client = texttospeech.TextToSpeechClient()
    # if not ssml:
    #     input_text = texttospeech.SynthesisInput(text=input)
    # else:
    #     input_text = texttospeech.SynthesisInput(ssml=input)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="da-DK",
        name="da-DK-Neural2-D",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    translated_audio_files = []
    for text_segment_list in speaker_text_dict.values():
        speaker_translated_audio_file_list = []
        for text in text_segment_list:
            input_text = texttospeech.SynthesisInput(text=text)
            response = client.synthesize_speech(
                request={"input": input_text, "voice": voice, "audio_config": audio_config}
            )
            speaker_translated_audio_file_list.append(response.audio_content)
        translated_audio_files.append(speaker_translated_audio_file_list)

    tts_dir = os.path.join(output_dir, "speaker_tts")
    os.makedirs(tts_dir, exist_ok=True)
    durations = []
    for i, audio_file_list in enumerate(translated_audio_files):
        audio = AudioSegment.silent(duration=0)
        durations_speaker_i = []
        for audio_segment_file in audio_file_list:
            audio_segment = AudioSegment.from_mp3(BytesIO(audio_segment_file))
            audio += audio_segment
            durations_speaker_i.append(int(audio_segment.duration_seconds * 1000))
        durations.append(durations_speaker_i)
        output_path = os.path.join(tts_dir, f"speaker{i}.wav")
        audio.export(output_path, format="wav")

    return translated_audio_files, durations