#!/usr/bin/env python3
"""抓取 BRAIN 官方 API 的 FAQ / operators / tutorial-pages，
生成 brain_official_cache.json + brain_official_vectors.npy 供 kb_server fallback 使用。"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

ROOT = Path(__file__).parent
CRED = Path(os.getenv("BRAIN_CRED", "/Users/cony.zhangbjgmail.com/dev/wq-brain/brain_credentials.txt"))
CACHE_JSON = ROOT / "brain_official_cache.json"
CACHE_VECTORS = ROOT / "brain_official_vectors.npy"

API = "https://api.worldquantbrain.com"
ARK_EMB_API = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
ARK_EMB_MODEL = "doubao-embedding-vision-251215"
ARK_EMB_KEY = os.getenv("ARK_EMB_KEY", "")


def strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = s.replace("&#x27;", "'").replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def blocks_to_text(raw) -> str:
    """FAQ.answer / tutorial-pages.content 的 block 列表 → 纯文本。"""
    if isinstance(raw, str):
        try:
            blocks = json.loads(raw)
        except Exception:
            return raw
    else:
        blocks = raw
    out = []
    for b in blocks:
        t = b.get("type")
        v = b.get("value")
        if t == "TEXT" and isinstance(v, str):
            out.append(strip_html(v))
        elif t == "HEADING" and isinstance(v, dict):
            out.append(strip_html(v.get("content", "")))
        elif t == "CODE" and isinstance(v, str):
            out.append(f"```\n{v}\n```")
        elif t == "LIST" and isinstance(v, list):
            for item in v:
                out.append("- " + strip_html(str(item)))
    return "\n".join(filter(None, out)).strip()


def login() -> requests.Session:
    u, p = json.load(open(CRED))
    s = requests.Session()
    s.trust_env = False
    s.auth = HTTPBasicAuth(u, p)
    r = s.post(f"{API}/authentication", timeout=30)
    r.raise_for_status()
    return s


def fetch_all_faqs(s: requests.Session) -> list[dict]:
    out, url = [], f"{API}/faqs?limit=200"
    while url:
        j = s.get(url, timeout=20).json()
        out.extend(j["results"])
        url = j.get("next")
    return out


def fetch_operators(s: requests.Session) -> list[dict]:
    return s.get(f"{API}/operators", timeout=30).json()


def fetch_tutorial_pages(s: requests.Session) -> list[dict]:
    out = []
    t_list = s.get(f"{API}/tutorials", timeout=20).json()["results"]
    for t in t_list:
        for p in t["pages"]:
            try:
                detail = s.get(f"{API}/tutorial-pages/{p['id']}", timeout=20).json()
                detail["_tutorial_category"] = t.get("category", "")
                detail["_tutorial_id"] = t["id"]
                out.append(detail)
                time.sleep(0.2)
            except Exception as e:
                print(f"skip {p['id']}: {e}")
    return out


def to_chunks(faqs: list[dict], operators: list[dict], pages: list[dict]) -> list[dict]:
    chunks = []
    for f in faqs:
        ans = blocks_to_text(f.get("answer", ""))
        if not ans:
            continue
        chunks.append({
            "title": f"[FAQ/{f['category']}] {f['question']}",
            "content": ans,
            "source": "brain-official-faq",
            "source_id": f["id"],
        })
    for op in operators:
        desc = op.get("description", "").strip()
        defn = op.get("definition", "").strip()
        if not desc and not defn:
            continue
        chunks.append({
            "title": f"[Operator/{op['category']}] {op['name']}",
            "content": f"定义：{defn}\n说明：{desc}",
            "source": "brain-official-operator",
            "source_id": op["name"],
        })
    for pg in pages:
        body = blocks_to_text(pg.get("content", []))
        if not body:
            continue
        chunks.append({
            "title": f"[Tutorial/{pg.get('_tutorial_category', '')}] {pg['title']}",
            "content": body,
            "source": "brain-official-tutorial",
            "source_id": pg["id"],
        })
    return chunks


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    if not ARK_EMB_KEY:
        raise RuntimeError("ARK_EMB_KEY env var required")
    vectors = []
    for i, c in enumerate(chunks):
        text = f"{c['title']}\n{c['content']}"[:6000]
        resp = requests.post(
            ARK_EMB_API,
            headers={"Authorization": f"Bearer {ARK_EMB_KEY}", "Content-Type": "application/json"},
            json={"model": ARK_EMB_MODEL, "input": [{"type": "text", "text": text}]},
            timeout=30,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["data"]["embedding"])
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(chunks)}] {c['title'][:60]}")
        time.sleep(0.3)
    return np.array(vectors, dtype=np.float32)


def main():
    s = login()
    print("login OK")
    print("fetch FAQs...")
    faqs = fetch_all_faqs(s)
    print(f"  {len(faqs)} faqs")
    print("fetch operators...")
    operators = fetch_operators(s)
    print(f"  {len(operators)} operators")
    print("fetch tutorial pages...")
    pages = fetch_tutorial_pages(s)
    print(f"  {len(pages)} pages")
    chunks = to_chunks(faqs, operators, pages)
    print(f"total chunks: {len(chunks)}")
    print("embedding...")
    vecs = embed_chunks(chunks)
    CACHE_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(CACHE_VECTORS, vecs)
    print(f"saved: {CACHE_JSON} ({CACHE_JSON.stat().st_size} bytes)")
    print(f"saved: {CACHE_VECTORS} ({vecs.shape})")


if __name__ == "__main__":
    main()
