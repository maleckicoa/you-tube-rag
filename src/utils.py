import re
import json
from TTS.api import TTS


def extract_plaintext_from_json_subs(text):
        
    try:
        data = json.loads(text)
    except:
        return ""  # Not JSON subtitles → ignore

    result = []

    for event in data.get("events", []):
        for seg in event.get("segs", []):
            if "utf8" in seg:
                result.append(seg["utf8"])

    return re.sub(r"\s+", " ", "".join(result)).strip()
    

def extract_metadata(docs):
    return [
        {
            "title": doc.metadata.get("title"),
            "url": doc.metadata.get("url"),
            "id": doc.metadata.get("id"),
        }
        for doc in docs
    ] if docs else []


def format_context(docs):
    return "\n\n".join(d.page_content for d in docs)

def trim_history(history):
    return history[-20*2:] #keep 20 latest exchanges




_tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

def generate_tts_audio_bytes(text: str) -> bytes:
    """
    Convert text to speech and return audio as raw bytes (WAV).
    """
    # Generate WAV bytes into an in-memory buffer
    wav_path = "temp_output.wav"
    _tts.tts_to_file(text=text, file_path=wav_path, progress_bar=False, verbose=False)

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    return audio_bytes


def chatbot_text_response(user_input: str) -> str:
    """
    Your chatbot logic (placeholder).
    Replace with your real pipeline / RAG / OpenAI call.
    """
    return f"{user_input}"