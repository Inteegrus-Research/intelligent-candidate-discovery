#!/usr/bin/env python3
"""
Phase 3 verification.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

base = Path("data/artifacts")

parq = base / "jd_intents.parquet"
embp = base / "jd_intent_embeddings.npy"
namep = base / "jd_intent_names.json"

print("exists parquet:", parq.exists())
print("exists embeddings:", embp.exists())
print("exists names:", namep.exists())

df = pd.read_parquet(parq)
emb = np.load(embp, allow_pickle=True)

with open(namep, "r", encoding="utf-8") as f:
    names = json.load(f)

print("intent rows:", len(df))
print("intent names:", names)
print("embedding shape:", emb.shape)

print("\nEmpty texts:")
print((df["intent_text"].fillna("").str.len() == 0).sum())

print("\nLength stats:")
print(df[["intent_char_len", "intent_word_len"]].describe())

print("\nIntents preview:")
for _, row in df.iterrows():
    print("=" * 80)
    print(row["intent_name"])
    print(row["intent_text"][:500])
