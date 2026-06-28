#!/usr/bin/env python3
"""
End‑to‑end integrity verification for the candidate ranking pipeline.

Validates:
- Artifact existence
- Row counts and shapes
- ID alignment across stages
- No merge collisions
- Required columns and value ranges
- Monotonic scoring
- Suspicious top‑1000 patterns
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/artifacts")

# Updated artifact paths
ARTIFACTS = {
    "ids":                       BASE / "candidate_ids.npy",
    "metadata":                  BASE / "candidate_metadata.parquet",
    "preliminary_features":      BASE / "candidate_preliminary_features.parquet",
    "text":                      BASE / "candidate_text.parquet",
    "jd_intents":                BASE / "jd_intents.parquet",
    "jd_intent_embeddings":      BASE / "jd_intent_embeddings.npy",
    "jd_intent_names":           BASE / "jd_intent_names.json",
    "embeddings":                BASE / "candidate_embeddings.npy",
    "embedding_names":           BASE / "candidate_embedding_names.json",
    "features":                  BASE / "candidate_features.parquet",                       # after feature build
    "features_with_consistency": BASE / "candidate_features_with_consistency.parquet",
    "consistency_scores":        BASE / "candidate_consistency_scores.parquet",
    "top1000":                   BASE / "retrieval_top1000.parquet",
    "scored":                    BASE / "candidate_scores.parquet",
}


def must_exist(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def main():
    # ---- existence ----
    for name, path in ARTIFACTS.items():
        must_exist(path)

    # ---- load artifacts ----
    ids = np.load(ARTIFACTS["ids"], allow_pickle=True).astype(str)

    meta = pd.read_parquet(ARTIFACTS["metadata"])
    prep_feat = pd.read_parquet(ARTIFACTS["preliminary_features"])
    txt = pd.read_parquet(ARTIFACTS["text"])
    jd = pd.read_parquet(ARTIFACTS["jd_intents"])
    jd_emb = np.load(ARTIFACTS["jd_intent_embeddings"], allow_pickle=True)
    emb = np.load(ARTIFACTS["embeddings"], allow_pickle=True)
    features = pd.read_parquet(ARTIFACTS["features"])
    features_w_cons = pd.read_parquet(ARTIFACTS["features_with_consistency"])
    cons = pd.read_parquet(ARTIFACTS["consistency_scores"])
    top = pd.read_parquet(ARTIFACTS["top1000"])
    scored = pd.read_parquet(ARTIFACTS["scored"])

    # ---- structural checks ----
    assert len(ids) == 100000, f"candidate_ids count mismatch: {len(ids)}"
    assert len(meta) == 100000, f"metadata rows mismatch: {len(meta)}"
    assert len(prep_feat) == 100000, f"preliminary feature rows mismatch: {len(prep_feat)}"
    assert len(txt) == 100000, f"text rows mismatch: {len(txt)}"
    assert len(features) == 100000, f"features rows mismatch: {len(features)}"
    assert len(features_w_cons) == 100000, f"features with consistency rows mismatch: {len(features_w_cons)}"
    assert len(cons) == 100000, f"consistency scores rows mismatch: {len(cons)}"
    assert len(jd) == 5, f"jd intent rows mismatch: {len(jd)}"
    assert jd_emb.shape == (5, 384), f"jd embedding shape mismatch: {jd_emb.shape}"
    assert emb.shape == (100000, 4, 384), f"candidate embedding shape mismatch: {emb.shape}"

    # ---- ID alignment across stages ----
    for label, frame in [
        ("metadata", meta),
        ("preliminary features", prep_feat),
        ("text", txt),
        ("features", features),
        ("features with consistency", features_w_cons),
        ("consistency scores", cons),
    ]:
        arr = frame["candidate_id"].astype(str).to_numpy()
        assert np.array_equal(ids, arr), f"candidate_id order mismatch in {label}"

    # ---- no merge collisions ----
    assert not any(c.endswith("_x") or c.endswith("_y") for c in features_w_cons.columns), "merge collision columns in features_with_consistency"
    assert "honeypot_penalty" in features_w_cons.columns, "honeypot_penalty missing"
    assert "consistency_score" in features_w_cons.columns, "consistency_score missing"

    # ---- required columns in retrieval_top1000 ----
    required = [
        "retrieval_score",
        "final_score",
        "product_company_ratio",
        "honeypot_penalty",
        "consistency_score",
        "profile_similarity",
        "skills_similarity",
        "career_similarity",
        "full_similarity",
        "intent_weighted_full_similarity",
        "semantic_alignment_score",
        "activity_recency_score",
        "response_speed_score",
    ]
    missing = [c for c in required if c not in top.columns]
    assert not missing, f"Missing required columns in retrieval_top1000: {missing}"

    # ---- NaN checks ----
    for name, frame, cols in [
        ("features", features, [
            "profile_similarity", "skills_similarity", "career_similarity", "full_similarity",
            "intent1_similarity", "intent2_similarity", "intent3_similarity", "intent4_similarity", "intent5_similarity",
            "product_company_ratio",
        ]),
        ("features_with_consistency", features_w_cons, ["consistency_score", "honeypot_penalty", "profile_consistency_score"]),
        ("retrieval_top1000", top, ["retrieval_score", "final_score", "product_company_ratio", "honeypot_penalty", "consistency_score"]),
    ]:
        na_rates = frame[cols].isna().mean()
        assert float(na_rates.max()) == 0.0, f"NaNs present in {name}: {na_rates[na_rates > 0].to_dict()}"

    # ---- value ranges ----
    for c in ["honeypot_penalty", "consistency_score", "product_company_ratio", "notice_score"]:
        if c in top.columns:
            lo, hi = float(top[c].min()), float(top[c].max())
            print(f"{c}: {lo:.6f} .. {hi:.6f}")

    # ---- monotonic score ----
    top = top.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    final_scores = top["final_score"].to_numpy()
    assert all(final_scores[i] >= final_scores[i + 1] for i in range(len(final_scores) - 1)), "final_score not monotonic"

    # ---- suspect flags ----
    top["low_product_company"] = (top["product_company_ratio"] <= 0.10).astype(int)
    top["high_honeypot"] = (top["honeypot_penalty"] >= 0.20).astype(int)
    top["low_consistency"] = (top["consistency_score"] < 0.50).astype(int)
    top["inactive"] = (top.get("open_to_work", 0) == 0).astype(int) if "open_to_work" in top.columns else 0

    print("\n=== Pipeline summary ===")
    print(f"Top1000 rows: {len(top)}")
    print(f"Unique IDs in top1000: {top['candidate_id'].nunique()}")
    print(f"Duplicates in top1000: {top['candidate_id'].duplicated().sum()}")

    print("\n=== Suspect distribution in top1000 ===")
    print("Mean product_company_ratio (full):", float(features_w_cons["product_company_ratio"].mean()))
    print("Mean product_company_ratio (top1000):", float(top["product_company_ratio"].mean()))
    print("Share low product_company_ratio (<=0.10):", float(top["low_product_company"].mean()))
    print("Share high honeypot_penalty (>=0.20):", float(top["high_honeypot"].mean()))
    print("Share low consistency (<0.50):", float(top["low_consistency"].mean()))

    if float(top["product_company_ratio"].mean()) < float(features_w_cons["product_company_ratio"].mean()):
        print("WARNING: top1000 product-company ratio is not better than full pool.")

    print("\nTop 20 suspicious top1000 rows:")
    cols_show = [
        "candidate_id",
        "final_score",
        "retrieval_score",
        "product_company_ratio",
        "honeypot_penalty",
        "consistency_score",
    ]
    if "open_to_work" in top.columns:
        cols_show.append("open_to_work")
    if "days_active" in top.columns:
        cols_show.append("days_active")
    suspicious = top.sort_values(
        by=["honeypot_penalty", "low_product_company", "low_consistency", "final_score"],
        ascending=[False, False, True, False],
        kind="mergesort",
    ).head(20)
    print(suspicious[cols_show].to_string(index=False))

    print("\n=== Top 10 ===")
    print(
        top[
            ["candidate_id", "final_score", "retrieval_score", "product_company_ratio", "honeypot_penalty", "consistency_score"]
        ].head(10).to_string(index=False)
    )

    print("\nAll end-to-end checks passed.")


if __name__ == "__main__":
    main()
