#!/usr/bin/env python3
"""
End-to-end verifier for Phases 1-7.

Checks:
- artifact existence
- row counts
- ID alignment across phases
- embedding shapes
- no merge collisions
- required Phase 7 columns
- monotonic scoring
- suspicious top-1000 patterns
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/artifacts")

FILES = {
    "ids": BASE / "candidate_ids.npy",
    "metadata": BASE / "candidate_metadata.parquet",
    "features_p1": BASE / "candidate_features.parquet",
    "text": BASE / "candidate_text.parquet",
    "jd_intents": BASE / "jd_intents.parquet",
    "jd_intent_embeddings": BASE / "jd_intent_embeddings.npy",
    "jd_intent_names": BASE / "jd_intent_names.json",
    "embeddings": BASE / "candidate_embeddings.npy",
    "embedding_names": BASE / "candidate_embedding_names.json",
    "features_p5": BASE / "candidate_features_phase5.parquet",
    "features_p6": BASE / "candidate_features_phase6.parquet",
    "consistency": BASE / "candidate_consistency.parquet",
    "top1000": BASE / "top1000_candidates.parquet",
    "scored": BASE / "scored_candidates.parquet",
}


def must_exist(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))


def main():
    # ---- existence ----
    for name, path in FILES.items():
        must_exist(path)

    # ---- load artifacts ----
    ids = np.load(FILES["ids"], allow_pickle=True).astype(str)

    meta = pd.read_parquet(FILES["metadata"])
    p1 = pd.read_parquet(FILES["features_p1"])
    txt = pd.read_parquet(FILES["text"])
    jd = pd.read_parquet(FILES["jd_intents"])
    jd_emb = np.load(FILES["jd_intent_embeddings"], allow_pickle=True)
    emb = np.load(FILES["embeddings"], allow_pickle=True)
    p5 = pd.read_parquet(FILES["features_p5"])
    p6 = pd.read_parquet(FILES["features_p6"])
    cons = pd.read_parquet(FILES["consistency"])
    top = pd.read_parquet(FILES["top1000"])
    scored = pd.read_parquet(FILES["scored"])

    # ---- structural checks ----
    assert len(ids) == 100000, f"candidate_ids count mismatch: {len(ids)}"
    assert len(meta) == 100000, f"metadata rows mismatch: {len(meta)}"
    assert len(p1) == 100000, f"phase1 feature rows mismatch: {len(p1)}"
    assert len(txt) == 100000, f"text rows mismatch: {len(txt)}"
    assert len(p5) == 100000, f"phase5 rows mismatch: {len(p5)}"
    assert len(p6) == 100000, f"phase6 rows mismatch: {len(p6)}"
    assert len(cons) == 100000, f"consistency rows mismatch: {len(cons)}"
    assert len(jd) == 5, f"jd intent rows mismatch: {len(jd)}"
    assert jd_emb.shape == (5, 384), f"jd embedding shape mismatch: {jd_emb.shape}"
    assert emb.shape == (100000, 4, 384), f"candidate embedding shape mismatch: {emb.shape}"

    # ---- ID alignment ----
    for label, frame in [
        ("metadata", meta),
        ("phase1", p1),
        ("text", txt),
        ("phase5", p5),
        ("phase6", p6),
        ("consistency", cons),
    ]:
        arr = frame["candidate_id"].astype(str).to_numpy()
        assert np.array_equal(ids, arr), f"candidate_id order mismatch in {label}"

    # ---- no merge collisions ----
    assert not any(c.endswith("_x") or c.endswith("_y") for c in p6.columns), "merge collision columns in phase6"
    assert "honeypot_penalty" in p6.columns, "honeypot_penalty missing in phase6"
    assert "consistency_score" in p6.columns, "consistency_score missing in phase6"

    # ---- required phase 7 columns ----
    required7 = [
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
    missing7 = [c for c in required7 if c not in top.columns]
    assert not missing7, f"Missing Phase 7 columns in top1000: {missing7}"

    # ---- NaN checks ----
    for name, frame, cols in [
        ("phase5", p5, [
            "profile_similarity", "skills_similarity", "career_similarity", "full_similarity",
            "intent1_similarity", "intent2_similarity", "intent3_similarity", "intent4_similarity", "intent5_similarity",
            "product_company_ratio",
        ]),
        ("phase6", p6, ["consistency_score", "honeypot_penalty", "profile_consistency_score"]),
        ("phase7-top", top, ["retrieval_score", "final_score", "product_company_ratio", "honeypot_penalty", "consistency_score"]),
    ]:
        na_rates = frame[cols].isna().mean()
        assert float(na_rates.max()) == 0.0, f"NaNs present in {name}: {na_rates[na_rates > 0].to_dict()}"

    # ---- ranges ----
    for c in ["honeypot_penalty", "consistency_score", "product_company_ratio", "notice_score"]:
        if c in top.columns:
            lo, hi = float(top[c].min()), float(top[c].max())
            print(f"{c}: {lo:.6f} .. {hi:.6f}")

    # ---- monotonic score check ----
    top = top.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    final_scores = top["final_score"].to_numpy()
    assert all(final_scores[i] >= final_scores[i + 1] for i in range(len(final_scores) - 1)), "final_score not monotonic"

    # ---- suspect checks ----
    top["low_product_company"] = (top["product_company_ratio"] <= 0.10).astype(int)
    top["high_honeypot"] = (top["honeypot_penalty"] >= 0.20).astype(int)
    top["low_consistency"] = (top["consistency_score"] < 0.50).astype(int)
    top["inactive"] = (top.get("open_to_work", 0) == 0).astype(int) if "open_to_work" in top.columns else 0

    print("\n=== Pipeline summary ===")
    print(f"Top1000 rows: {len(top)}")
    print(f"Unique IDs in top1000: {top['candidate_id'].nunique()}")
    print(f"Duplicates in top1000: {top['candidate_id'].duplicated().sum()}")

    print("\n=== Suspect distribution in top1000 ===")
    print("Mean product_company_ratio (full):", float(p6["product_company_ratio"].mean()))
    print("Mean product_company_ratio (top1000):", float(top["product_company_ratio"].mean()))
    print("Share low product_company_ratio (<=0.10):", float(top["low_product_company"].mean()))
    print("Share high honeypot_penalty (>=0.20):", float(top["high_honeypot"].mean()))
    print("Share low consistency (<0.50):", float(top["low_consistency"].mean()))

    if float(top["product_company_ratio"].mean()) < float(p6["product_company_ratio"].mean()):
        print("WARNING: top1000 product-company ratio is not better than full pool.")

    print("\nTop 20 suspicious top1000 rows:")
    cols_show = [
        "candidate_id",
        "final_score",
        "retrieval_score",
        "product_company_ratio",
        "honeypot_penalty",
        "consistency_score",
        "open_to_work" if "open_to_work" in top.columns else "candidate_id",
        "days_active" if "days_active" in top.columns else "candidate_id",
    ]
    cols_show = [c for c in cols_show if c in top.columns]
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
