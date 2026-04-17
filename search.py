import json
import os

DOC_PATH = "./storage/doecment.json"

def search_chunks(keyword: str, topn: int = 20):
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    hits = []
    for idx, text in enumerate([i['text'] for i in docs]):
        if keyword in text:
            hits.append((idx, text))

    print(f"Found {len(hits)} chunks containing: {keyword}")
    for idx, text in hits[:topn]:
        preview = text.replace("\n", " ")[:200]
        print(f"chunk_id={idx}  preview={preview}")

search_chunks("加权平均成绩")
search_chunks("推免程序")