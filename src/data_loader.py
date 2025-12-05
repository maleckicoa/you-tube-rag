from yt_dlp import YoutubeDL
import os
import json
import uuid
import argparse

from dotenv import load_dotenv
from zipfile import ZipFile, ZIP_DEFLATED
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import warnings
warnings.filterwarnings("ignore")

from urllib.request import urlopen
from src.utils import extract_plaintext_from_json_subs

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_playlist_captions(playlist_url):


    thumbs_dir = "data/thumbnails"
    os.makedirs(thumbs_dir, exist_ok=True)

    opts = {
        "skip_download": True,
        "writeautomaticsub": True,
        "subtitlesformat": "srt", 
        "writesubtitles": True, 
        "convert_subtitles": "srt",
        "quiet": True,
        "ignoreerrors": True,
        "no_warnings": True,
        "cookiesfrombrowser": ("chrome",),
        }

    results = []

    with YoutubeDL(opts) as ydl:
        playlist = ydl.extract_info(playlist_url, download=False)

        for entry in playlist["entries"]:
            if not entry:
                continue

            video_url = entry["webpage_url"]
            title = entry["title"]
            video_id = entry.get("id")  # YouTube video id if available
            
            # Build thumbnail URL and download to data/thumbnails
            thumbnail_url = None
            if video_id:
                thumbnail_url = f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                try:
                  
                    thumb_bytes = urlopen(thumbnail_url).read()
                    thumb_path = thumbs_dir + f"/{video_id}.jpg"
   
                    with open(thumb_path, "wb") as tf:
                        tf.write(thumb_bytes)
                    thumbnail_path_rel = f"data/thumbnails/{video_id}.jpg"
                except Exception:
                    thumbnail_path_rel = None
            else:
                thumbnail_path_rel = None

            
            subs = entry.get("subtitles") or entry.get("automatic_captions") or {} # Extract subtitles
            en = subs.get("en") or subs.get("en-US") or []                         # Pick English if exists
            caption_text = ""

            # Download each caption track as text
            for track in en:
                if "url" in track:
                    cap = ydl.urlopen(track["url"]).read().decode("utf8")
                    caption_text += extract_plaintext_from_json_subs(cap) #+ "\n"

            results.append({
                "id": str(uuid.uuid4()),
                "video_id": video_id,
                "url": video_url,
                "title": title,
                "text": caption_text.strip(),
                "thumbnail": thumbnail_path_rel,
            })


    results = [item for item in results if item["text"] != ""]
    output_json = "data/playlist_captions.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Done → {output_json}")

    # Zip the data directory
    zip_path = Path("data.zip")
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as z:
        for root, _, files in os.walk("data"):
            for name in files:
                fp = Path(root) / name          
                z.write(fp, arcname=fp)        
    print("Zipped →", zip_path)



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




DEFAULT_PLAYLISTS = {
    "full": "https://www.youtube.com/playlist?list=PLD18sR-9Y-XFVCP-cSjCLr8A1B_IRsvp-",
    "short": "https://www.youtube.com/playlist?list=PLD18sR-9Y-XH4WXRV9aSLKTU9a_eh6j6_",
}


def _resolve_playlist(p: str | None) -> str:
    if not p:
        return DEFAULT_PLAYLISTS["full"]
    return DEFAULT_PLAYLISTS.get(p, p)  # name → URL, or accept raw URL




def main():
    parser = argparse.ArgumentParser(prog="data_loader", description="YouTube captions → JSON → Chroma")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # create: accepts an optional name (full/short) or a raw URL
    p_create = sub.add_parser("create", help="Create data/playlist_captions.json + thumbnails")
    p_create.add_argument("playlist", nargs="?", help="playlist name (full/short) or full URL")

    # embed: embeds JSON (default path)
    p_embed = sub.add_parser("embed", help="Embed captions JSON into Chroma")
    p_embed.add_argument("--captions", default="data/playlist_captions.json")

    # all: create then embed
    p_all = sub.add_parser("all", help="Create then embed")
    p_all.add_argument("playlist", nargs="?", help="playlist name (full/short) or full URL")
    p_all.add_argument("--captions", default="data/playlist_captions.json")

    args = parser.parse_args()

    if args.cmd == "create":
        url = _resolve_playlist(args.playlist)
        create_playlist_captions(playlist_url=url)
    elif args.cmd == "embed":
        embed_playlist_captions(captions=args.captions, API_KEY=OPENAI_API_KEY)
    elif args.cmd == "all":
        url = _resolve_playlist(args.playlist)
        create_playlist_captions(playlist_url=url)
        embed_playlist_captions(captions=args.captions, API_KEY=OPENAI_API_KEY)

if __name__ == "__main__":
    main()    

