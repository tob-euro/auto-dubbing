import deepl
import os

def translate(text: str, source_lang: str, target_lang: str, auth_key: str,
              processed_folder: str = None, video_base: str = None) -> str:
    """
    Translate the given text from source_lang to target_lang using DeepL.
    
    If source_lang is "AUTO" (case-insensitive), the translator will
    auto-detect the source language.
    
    Args:
        text (str): Text to translate.
        source_lang (str): Source language code, or "AUTO" to auto-detect.
        target_lang (str): Target language code (e.g., "EN-US", "DE").
        auth_key (str): Your DeepL API authentication key.
        processed_folder (str, optional): Root folder for processed outputs.
        video_base (str, optional): Base name of the video for naming output file.
        
    Returns:
        str: The translated text.
    """
    translator = deepl.Translator(auth_key)
    
    if source_lang.upper() == "AUTO":
        result = translator.translate_text(text, target_lang=target_lang)
    else:
        result = translator.translate_text(text, source_lang=source_lang, target_lang=target_lang)
    
    translated_text = result.text
    
    # Save the translated text if output path info is provided.
    if processed_folder and video_base:
        # Save directly under the video folder.
        output_dir = os.path.join(processed_folder, video_base)
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"translation_{video_base}.txt"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated_text)
        print(f"Translation saved to: {output_path}")
    
    return translated_text

if __name__ == "__main__":
    # Test the api key works
    from dotenv import load_dotenv
    load_dotenv()
    deepl_key = os.getenv("DEEPL_API_KEY")
    test_text = "Hello, world!"
    translated_text = translate(
        text=test_text,
        source_lang="AUTO",
        target_lang="DA",
        auth_key=deepl_key,
    )
    print(f"Translated text: {translated_text}")