from yt_dlp import YoutubeDL
import json
import uuid

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import os
from utils import extract_plaintext_from_json_subs
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_playlist_captions(playlist_url):

    #playlist_url = "https://www.youtube.com/playlist?list=PLD18sR-9Y-XG8OS7CZ_3IQCux5KzgcCiZ"

    opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitlesformat": "srt", 
        "writesubtitles": True, 
        "convert_subtitles": "srt",
        "quiet": True,
        "ignoreerrors": True,
    }

    results = []

    with YoutubeDL(opts) as ydl:
        playlist = ydl.extract_info(playlist_url, download=False)

        for entry in playlist["entries"]:
            if not entry:
                continue

            video_url = entry["webpage_url"]
            title = entry["title"]

            # Extract subtitles (if available)
            subs = entry.get("subtitles") or entry.get("automatic_captions") or {}

            # Pick English if exists
            en = subs.get("en") or subs.get("en-US") or []
            caption_text = ""

            # Download each caption track as text
            for track in en:
                if "url" in track:
                    cap = ydl.urlopen(track["url"]).read().decode("utf8")
                    caption_text += extract_plaintext_from_json_subs(cap) #+ "\n"

            results.append({
                "id": str(uuid.uuid4()),
                "url": video_url,
                "title": title,
                "text": caption_text.strip()
            })



    results = [item for item in results if item["text"] != ""]
    with open("playlist_captions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done → playlist_captions.json")




def embed_playlist_captions(captions, API_KEY):

    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=API_KEY)
    
    json_path = Path(captions)
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)


    docs = [
        Document(
            page_content=item["text"],
            metadata={"id": item["id"], "title": item["title"], "url": item["url"]}
        )
        for item in results
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(docs)

    

    Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        collection_name="youtube_captions",
        persist_directory="chroma_db"
    )
    print("DOCUMENT COUNT IN CHROMA:", Chroma(collection_name="youtube_captions", persist_directory="chroma_db")._collection.count())


if __name__ == "__main__":
    create_playlist_captions(playlist_url = "https://www.youtube.com/playlist?list=PLD18sR-9Y-XFVCP-cSjCLr8A1B_IRsvp-")
    embed_playlist_captions(captions = "playlist_captions.json", API_KEY = OPENAI_API_KEY)