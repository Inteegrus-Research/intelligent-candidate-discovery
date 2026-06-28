#!/usr/bin/env python3
"""
Build the main feature table by joining semantic, skill, career,
behavioral, availability, and quality features.

Inputs:
- data/raw/candidates.jsonl
- data/artifacts/candidate_ids.npy
- data/artifacts/candidate_preliminary_features.parquet
- data/artifacts/candidate_metadata.parquet
- data/artifacts/candidate_embeddings.npy
- data/artifacts/jd_intent_embeddings.npy

Outputs:
- data/artifacts/candidate_features.parquet
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from features.common import load_raw_candidates
from features.semantic_features import compute_semantic_features
from features.skill_features import extract_skill_features
from features.career_features import extract_career_features
from features.behavioral_features import extract_behavioral_features
from features.availability_features import extract_availability_features
from features.quality_features import extract_quality_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/candidates.jsonl")
    ap.add_argument("--candidate-ids", default="data/artifacts/candidate_ids.npy")
    ap.add_argument("--base-features", default="data/artifacts/candidate_preliminary_features.parquet")
    ap.add_argument("--metadata", default="data/artifacts/candidate_metadata.parquet")
    ap.add_argument("--candidate-embeddings", default="data/artifacts/candidate_embeddings.npy")
    ap.add_argument("--jd-embeddings", default="data/artifacts/jd_intent_embeddings.npy")
    ap.add_argument("--out", default="data/artifacts/candidate_features.parquet")
    ap.add_argument("--as-of", default="2026-06-20")
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of)

    candidate_ids = np.load(args.candidate_ids, allow_pickle=True)
    base = pd.read_parquet(args.base_features)
    meta = pd.read_parquet(args.metadata)[["candidate_id", "location", "country"]]

    # Clean and fill missing flags
    base["github_activity_missing"] = (base["github_activity"] == -1).astype(int)
    base["offer_acceptance_missing"] = (base["offer_acceptance"] == -1).astype(int)
    base["github_activity"] = base["github_activity"].where(base["github_activity"] != -1, 0.0)
    base["offer_acceptance"] = base["offer_acceptance"].where(base["offer_acceptance"] != -1, 0.5)
    base["days_active"] = base["days_since_active"]
    base["profile_views"] = base["views_received_30d"]
    base["saved_by_recruiters"] = base["saved_by_recruiters_30d"]
    base["search_appearance"] = base["search_appearance_30d"]
    base["product_company_ratio"] = np.where(
        base["career_count"] > 0, 1.0 - base["service_company_ratio"].fillna(0), 0.0
    )

    sem = compute_semantic_features(args.candidate_embeddings, args.jd_embeddings, candidate_ids)

    rows = []
    for i, c in enumerate(load_raw_candidates(args.data)):
        cid = c["candidate_id"]
        assert str(candidate_ids[i]) == cid, f"Order mismatch at row {i}"

        prof = c["profile"]
        sig = c["redrob_signals"]
        skills = c.get("skills", [])
        career = c.get("career_history", [])

        base_row = base.iloc[i].to_dict()

        row = {"candidate_id": cid}
        row.update(extract_skill_features(skills, sig.get("skill_assessment_scores", {})))
        row.update(extract_career_features(career, prof))
        row.update(extract_behavioral_features(sig, as_of))
        row.update(extract_availability_features(sig, prof))
        row.update(extract_quality_features(base_row, sig, skills, career))
        rows.append(row)

    extra = pd.DataFrame(rows)
    final = base.merge(meta, on="candidate_id", how="left").merge(sem, on="candidate_id", how="left")
    final = final.merge(extra, on="candidate_id", how="left", suffixes=("", "_new"))

    overwrite_cols = [
        "response_rate", "interview_completion", "open_to_work",
        "verified_email", "verified_phone", "linkedin_connected",
        "willing_to_relocate", "github_activity", "offer_acceptance",
        "notice_period_days",
    ]
    for col in overwrite_cols:
        if f"{col}_new" in final.columns:
            final[col] = final[f"{col}_new"]
            final.drop(columns=[f"{col}_new"], inplace=True)

    drop_cols = [c for c in final.columns if c.endswith("_new")]
    if drop_cols:
        final.drop(columns=drop_cols, inplace=True)

    numeric_cols = final.select_dtypes(include=["number", "float", "int"]).columns
    final[numeric_cols] = final[numeric_cols].replace([np.inf, -np.inf], np.nan)

    out = Path(args.out)
    final.to_parquet(out, index=False)

    print(f"Feature table built. Saved: {out}")
    print("Rows:", len(final))
    print("Columns:", len(final.columns))


if __name__ == "__main__":
    main()
