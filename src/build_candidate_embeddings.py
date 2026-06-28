#!/usr/bin/env python3
"""
Generate embeddings for candidate documents.

Inputs:
- data/artifacts/candidate_text.parquet

Outputs:
- data/artifacts/candidate_embeddings.npy  (N, 4, 384)
- data/artifacts/candidate_embedding_names.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from retrieval.embedder import embed_texts


DOC_COLUMNS = ["profile_doc", "skills_doc", "career_doc", "full_doc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-path", default="data/artifacts/candidate_text.parquet")
    ap.add_argument("--outdir", default="data/artifacts")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    text_path = Path(args.text_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(text_path)
    missing_cols = [c for c in DOC_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required document columns: {missing_cols}")

    n = len(df)
    d = 384
    all_embs = np.zeros((n, len(DOC_COLUMNS), d), dtype=np.float16)

    for j, col in enumerate(DOC_COLUMNS):
        texts = df[col].fillna("").astype(str).tolist()
        print(f"Embedding {col} ...")
        emb = embed_texts(
            texts,
            model_name=args.model,
            batch_size=args.batch_size,
            normalize=True,
        )
        if emb.shape != (n, d):
            raise ValueError(f"Unexpected shape for {col}: {emb.shape}")
        all_embs[:, j, :] = emb

    emb_path = outdir / "candidate_embeddings.npy"
    np.save(emb_path, all_embs)

    names_path = outdir / "candidate_embedding_names.json"
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(DOC_COLUMNS, f, indent=2)

    print("Candidate embeddings built.")
    print(f"Saved: {emb_path}")
    print(f"Saved: {names_path}")
    print("Embedding tensor shape:", all_embs.shape)
    print("dtype:", all_embs.dtype)


if __name__ == "__main__":
    main()
