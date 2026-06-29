#!/usr/bin/env python3
"""用知识库回答 10 道模拟题。"""
import json
import os
from pathlib import Path

import numpy as np
import requests

INDEX_PATH = Path(__file__).parent / "kb_index.json"
VECTORS_PATH = Path(__file__).parent / "kb_vectors.npy"

ARK_EMB_API = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
ARK_EMB_MODEL = "doubao-embedding-vision-251215"
ARK_EMB_KEY = os.getenv("ARK_EMB_KEY", "")

ARK_CHAT_API = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
ARK_CHAT_MODEL = "doubao-1-5-pro-32k-250115"
ARK_API_KEY = os.getenv("ARK_API_KEY", "")


def load_index():
    with open(INDEX_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    vectors = np.load(VECTORS_PATH)
    return chunks, vectors


def get_embedding(text: str) -> np.ndarray:
    resp = requests.post(
        ARK_EMB_API,
        headers={
            "Authorization": f"Bearer {ARK_EMB_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": ARK_EMB_MODEL,
            "input": [{"type": "text", "text": text}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return np.array(resp.json()["data"]["embedding"], dtype=np.float32)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(question: str, chunks, vectors, top_k=3):
    q_emb = get_embedding(question)
    sims = [(cosine_similarity(q_emb, vec), i) for i, vec in enumerate(vectors)]
    sims.sort(reverse=True)
    contexts = []
    for sim, idx in sims[:top_k]:
        chunk = chunks[idx]
        contexts.append(f"【{chunk['title']}】\n{chunk['content']}")
    return "\n\n---\n\n".join(contexts)


def answer(question: str, context: str) -> str:
    system = (
        "你是一个 BRAIN Consultant 考试助手。根据下面的知识库内容，"
        "回答用户的问题。只使用知识库中的信息，不要编造。"
        "如果知识库中没有相关信息，请直接说明。"
    )
    prompt = f"知识库内容：\n\n{context}\n\n用户问题：{question}"
    resp = requests.post(
        ARK_CHAT_API,
        headers={
            "Authorization": f"Bearer {ARK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": ARK_CHAT_MODEL,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


QUESTIONS = [
    "Consultant 考试通过的 Sharpe 比率最低要求是多少？选项：A. 0.5 B. 0.7 C. 1.0 D. 1.25",
    "以下哪个 operator 用于计算过去 N 天的排名？选项：A. ts_mean(x, d) B. ts_rank(x, d) C. rank(x) D. ts_zscore(x, d)",
    "POST /submit 返回 201/200 即表示 alpha 已成功提交上线，不需要再通过 /alphas/{id} 查询状态验证。正确还是错误？",
    "写出一个 Alpha 表达式，计算 volume 的 10 日简单移动平均。",
    "以下哪些字段属于 fundamental（基本面）数据？选项：A. volume B. equity C. cashflow_op D. close",
    "/check 端点返回 alpha 状态为 SELF_CORRELATION，此时最可靠的做法是什么？选项：A. 直接调用 /alphas/{id} 查询 B. 调用 /check 端点重新查询 C. 等待 5 分钟后直接 submit D. 放弃该 alpha 重新写",
    "写出计算 (equity + cashflow_op) / cap 在过去 20 天内排名的完整表达式。",
    "BRAIN 积分体系中，触发顾问邀请（Advisor Invitation）的积分门槛是多少？选项：A. 5000 分 B. 7500 分 C. 10000 分 D. 15000 分",
    "Alpha 的 turnover（换手率）越高，通常意味着策略的交易频率越高、交易成本越大。正确还是错误？",
    "Fitness 指标主要由哪三个维度综合计算得出？",
]


def main():
    chunks, vectors = load_index()
    print(f"已加载知识库: {len(chunks)} chunks, vectors={vectors.shape}")
    print("=" * 60)

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n【第 {i} 题】{q}")
        print("-" * 40)
        try:
            ctx = search(q, chunks, vectors, top_k=3)
            ans = answer(q, ctx)
            print(f"知识库回答：{ans}")
        except Exception as e:
            print(f"查询出错：{e}")
        print("=" * 60)


if __name__ == "__main__":
    main()
