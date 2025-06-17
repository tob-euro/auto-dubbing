# 🎙️ Auto-Dubbing

A pipeline for automatically dubbing videos in any langauge supported by [AssemblyAI](https://www.assemblyai.com/docs/speech-to-text/pre-recorded-audio/speaker-diarization#supported-languages) into any language supported by both [DeepL](https://support.deepl.com/hc/en-us/articles/360019925219-DeepL-Translator-languages) and [Google Cloud](https://cloud.google.com/text-to-speech/docs/list-voices-and-types), which is almost any language.

## 🚀 Features

- Source seperation using Demucs
- Transcription using stable Whisper (speech-to-text)
- Machine translation using DeepL (text-to-text)
- Text-to-speech (tts) using Google-Cloud
- Voice conversion using seed-vc (speech-to-speech)

## 📦 Installation
This guide requires you to have installed [conda](https://www.anaconda.com/) and [ffmpeg](https://ffmpeg.org/) on your pc. You can test this by running "conda" and "ffmpeg" in a terminal without getting an error.

Clone the repository to your local machine and install the requirements in an environment using python 3.11
```bash
git clone https://github.com/tob-euro/auto-dubbing
cd auto-dubbing
conda create -n AutoDubbing python=3.11
conda activate AutoDubbing
pip install -e .
pip install -r requirements.txt
conda deactivate
```

Create data directory with input folder
```bash
mkdir data\input
```

Clone the seed-vc repository in the root directory. Create a separate python environment (3.10.11), that executes the seed-vc model.
```bash
git clone https://github.com/Plachtaa/seed-vc
cd seed-vc
conda create -n seed-vc python=3.10.11
conda activate seed-vc
pip install wheel
pip install -r requirements.txt
conda deactivate
```

Get API keys from [DeepL](https://www.deepl.com) and [AssemblyAI](https://www.assemblyai.com/).

Set up a [Google Cloud](https://console.cloud.google.com) project, enable "Cloud Text-to-Speech API" under APIs & Services. You have to enable billing for your project in order to do this, so you have to set that up. You can get free credits with a free trial, and in any case it should not cost anything since we use a Neural2 voice, which is free the for the first 1 million characters. Create a service account with Editor role access, then create a new key for this service account and download the JSON file. Rename it to "gcp_key.json" and place in root directory.

Create a .env file in the root directory containing the following varaibles:

```bash
DEEPL_API_KEY="YOUR_DEEPL_API_KEY"
ASSEMBLY_API_KEY="YOUR_ASSEMBLY_API_KEY"
GOOGLE_APPLICATION_CREDENTIALS="gcp_key.json"
SEED_VC_PYTHON_PATH="PATH_TO_SEED_VC_PYTHON_EXECUTABLE"
```

Insert your two API keys, google cloud credentials JSON file and the path to your python executable in your seed-vc conda environment.

## 📁File structute
Your file structure should look like this

```bash
auto-dubbing/
├── data/
│    └── input/
├── seed-vc/
├── src/
│   └── auto_dubbing/
│       ├── transcription.py
│       ├── mixing.py
│       └── tts.py
├── .env
├── .gitignore
├── config.yaml
├── gcp_key.json
├── pyproject.toml
├── README.md
├── requirements.txt
└── run.py
```

## 🛠️Usage
Download a video file you want to dub. Name the video file (mp4) in the format "video_x.mp4" where x is a natural number, and put it in the data/input folder. Navigate to the config file "config.yaml", and change the "input_video" to match the name you gave the video:

```bash
input_video: "${paths.input_folder}/video_x.mp4"
```

Run the pipeline by running run.py using the AutoDubbing intepreter or use command line:

```bash
python run.py
```