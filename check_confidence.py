#!/usr/bin/env python3
"""检查3道"不会"的题，看看检索相似度是否偏低。"""
import json
import numpy as np
import requests
import os

INDEX_PATH = "kb_index.json"
VECTORS_PATH = "kb_vectors.npy"
ARK_EMB_API = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
ARK_EMB_KEY = os.getenv("ARK_EMB_KEY", "")

with open(INDEX_PATH) as f:
    chunks = json.load(f)
vectors = np.load(VECTORS_PATH)

def get_emb(text):
    resp = requests.post(ARK_EMB_API, headers={"Authorization": f"Bearer {ARK_EMB_KEY}", "Content-Type": "application/json"},
                         json={"model": "doubao-embedding-vision-251215", "input": [{"type": "text", "text": text}]}, timeout=30)
    return np.array(resp.json()["data"]["embedding"], dtype=np.float32)

def sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

questions = [
    "POST /submit 返回 201/200 即表示 alpha 已成功提交上线，不需要再通过 /alphas/{id} 查询状态验证。正确还是错误？",
    "写出一个 Alpha 表达式，计算 volume 的 10 日简单移动平均。",
    "/check 端点返回 alpha 状态为 SELF_CORRELATION，此时最可靠的做法是什么？选项：A. 直接调用 /alphas/{id} 查询 B. 调用 /check 端点重新查询 C. 等待 5 分钟后直接 submit D. 放弃该 alpha 重新写",
]

print("检查3道盲区题的检索相似度：\n")
for q in questions:
    q_emb = get_emb(q)
    sims = sorted([(sim(q_emb, vec), i) for i, vec in enumerate(vectors)], reverse=True)
    print(f"Q: {q[:40]}...")
    print(f"  top-1 sim={sims[0][0]:.4f} | {chunks[sims[0][1]]['title'][:50]}")
    print(f"  top-2 sim={sims[1][0]:.4f} | {chunks[sims[1][1]]['title'][:50]}")
    print(f"  top-3 sim={sims[2][0]:.4f} | {chunks[sims[2][1]]['title'][:50]}")
    print()

# 作为对比，检查一道会的题
q_ok = "Consultant 考试通过的 Sharpe 比率最低要求是多少？"
q_emb = get_emb(q_ok)
sims = sorted([(sim(q_emb, vec), i) for i, vec in enumerate(vectors)], reverse=True)
print(f"对比——会的题: {q_ok[:40]}...")
print(f"  top-1 sim={sims[0][0]:.4f} | {chunks[sims[0][1]]['title'][:50]}")
print(f"  top-2 sim={sims[1][0]:.4f} | {chunks[sims[1][1]]['title'][:50]}")
print(f"  top-3 sim={sims[2][0]:.4f} | {chunks[sims[2][1]]['title'][:50]}")
