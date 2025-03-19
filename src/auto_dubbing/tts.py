from google.cloud import texttospeech
import os
def synthesize_text(utterance_data, output_dir):
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
        audio_encoding=texttospeech.AudioEncoding.WAV
    )
    translated_audio_files = []
    for utterance in utterance_data:
        input_text = texttospeech.SynthesisInput(text=utterance["Translated_text"])
        response = client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        translated_audio_files.append(response.audio_content)

    for i, audio_file in enumerate(translated_audio_files):
        with open(os.path.join(output_dir, f"translated_segment{i}.wav"), "wb") as f:
            f.write(audio_file)
    # The response's audio_content is binary.
    return translated_audio_files
    # with open("output.mp3", "wb") as out:
    #     out.write(response.audio_content)
    #     print('Audio content written to file "output.mp3"')