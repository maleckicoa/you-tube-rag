import re
import json


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