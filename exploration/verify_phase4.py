#!/usr/bin/env python3
"""
Phase 4 verification.
"""

import json
from pathlib import Path

import numpy as np

base = Path("data/artifacts")
emb_path = base / "candidate_embeddings.npy"
name_path = base / "candidate_embedding_names.json"

print("exists embeddings:", emb_path.exists())
print("exists names:", name_path.exists())

emb = np.load(emb_path, allow_pickle=True)
print("shape:", emb.shape)
print("dtype:", emb.dtype)
print("min:", float(emb.min()))
print("max:", float(emb.max()))

with open(name_path, "r", encoding="utf-8") as f:
    names = json.load(f)

print("doc names:", names)

# quick sanity checks
print("candidate count:", emb.shape[0])
print("doc views:", emb.shape[1])
print("embedding dim:", emb.shape[2])

# check for NaN / inf
print("nan count:", np.isnan(emb).sum())
print("inf count:", np.isinf(emb).sum())

# show a tiny slice for human sanity
print("sample vector slice:", emb[0, 0, :10])
