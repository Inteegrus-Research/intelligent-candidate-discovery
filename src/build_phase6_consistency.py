#!/usr/bin/env python3
"""
Phase 6 (fixed correctly): build consistency scores using REAL skill details.

Inputs:
- data/raw/candidates.jsonl   (or .jsonl.gz)
- data/artifacts/candidate_features_phase5.parquet
- data/artifacts/candidate_text.parquet
- data/artifacts/candidate_metadata.parquet

Outputs:
- data/artifacts/candidate_consistency.parquet
- data/artifacts/candidate_features_phase6.parquet
"""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from scoring.consistency_engine import compute_consistency_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        default="data/raw/candidates.jsonl",
        help="Raw candidates file (jsonl or jsonl.gz)",
    )
    ap.add_argument(
        "--features",
        default="data/artifacts/candidate_features_phase5.parquet",
    )
    ap.add_argument(
        "--text",
        default="data/artifacts/candidate_text.parquet",
    )
    ap.add_argument(
        "--meta",
        default="data/artifacts/candidate_metadata.parquet",
    )
    ap.add_argument(
        "--out",
        default="data/artifacts/candidate_consistency.parquet",
    )
    ap.add_argument(
        "--merged-out",
        default="data/artifacts/candidate_features_phase6.parquet",
    )
    args = ap.parse_args()

    feat = pd.read_parquet(args.features)
    txt = pd.read_parquet(args.text)[["candidate_id", "profile_doc", "career_doc"]]
    meta = pd.read_parquet(args.meta)[["candidate_id", "current_title"]]

    # Load real raw skills once
    raw_path = Path(args.raw)
    if raw_path.suffix == ".gz":
        f = gzip.open(raw_path, "rt", encoding="utf-8")
    else:
        f = open(raw_path, "r", encoding="utf-8")

    skills_dict = {}
    with f:
        for line in f:
            c = json.loads(line.strip())
            skills_dict[c["candidate_id"]] = c.get("skills", [])

    df = feat.merge(txt, on="candidate_id", how="left").merge(meta, on="candidate_id", how="left")

    rows = []
    for _, r in df.iterrows():
        cid = r["candidate_id"]
        yoe = float(r.get("yoe", 0) or 0)
        career_years = float(r.get("career_years_from_history", 0) or 0)
        current_title = r.get("current_title", "")
        summary_text = r.get("profile_doc", "")
        career_text = r.get("career_doc", "")
        advanced = int(r.get("advanced_skill_count", 0) or 0)
        expert = int(r.get("expert_skill_count", 0) or 0)

        real_skills = skills_dict.get(cid, [])

        out = compute_consistency_score(
            yoe=yoe,
            career_years=career_years,
            skills=real_skills,
            current_title=current_title,
            summary_text=summary_text,
            career_text=career_text,
            advanced_skill_count=advanced,
            expert_skill_count=expert,
        )
        rows.append({"candidate_id": cid, **out})

    cons = pd.DataFrame(rows)
    cons.to_parquet(args.out, index=False)

    # Remove old columns from Phase 5 before merging to avoid _x/_y collisions
    feat = feat.drop(
        columns=[
            "consistency_score",
            "honeypot_penalty",
            "yoe_alignment_score",
            "domain_alignment_score",
            "skill_evidence_score",
        ],
        errors="ignore",
    )

    merged = feat.merge(cons, on="candidate_id", how="left")
    merged.to_parquet(args.merged_out, index=False)

    print("Phase 6 (fixed correctly) complete.")
    print(f"Saved: {args.out}")
    print(f"Saved merged features: {args.merged_out}")
    print(merged[["consistency_score", "honeypot_penalty"]].describe())


if __name__ == "__main__":
    main()
