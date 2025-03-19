from google.cloud import texttospeech
from pydub import AudioSegment
from io import BytesIO
import os

def synthesize_text(speaker_text_dict, output_dir):
    """Synthesizes speech from the input string of text."""
    
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
    for text in speaker_text_dict.values():
        input_text = texttospeech.SynthesisInput(text=text)
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        translated_audio_files.append(response.audio_content)

    tts_dir = os.path.join(output_dir, "speaker_tts")
    os.makedirs(tts_dir, exist_ok=True)

    for i, audio_file in enumerate(translated_audio_files):
        audio = AudioSegment.from_mp3(BytesIO(audio_file))
        output_path = os.path.join(tts_dir, f"speaker{i}.wav")
        audio.export(output_path, format="wav")

    return translated_audio_files