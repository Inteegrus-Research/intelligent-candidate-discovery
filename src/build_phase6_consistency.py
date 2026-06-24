#!/usr/bin/env python3
"""
Phase 6: build consistency scores for all candidates.

Inputs:
- data/artifacts/candidate_features_phase5.parquet
- data/artifacts/candidate_text.parquet
- data/artifacts/candidate_metadata.parquet

Output:
- data/artifacts/candidate_consistency.parquet
- data/artifacts/candidate_features_phase6.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from scoring.consistency_engine import compute_consistency_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/artifacts/candidate_features_phase5.parquet")
    ap.add_argument("--text", default="data/artifacts/candidate_text.parquet")
    ap.add_argument("--meta", default="data/artifacts/candidate_metadata.parquet")
    ap.add_argument("--out", default="data/artifacts/candidate_consistency.parquet")
    ap.add_argument("--merged-out", default="data/artifacts/candidate_features_phase6.parquet")
    args = ap.parse_args()

    feat = pd.read_parquet(args.features)
    txt = pd.read_parquet(args.text)
    meta = pd.read_parquet(args.meta)

    # Keep only the pieces needed for consistency scoring.
    txt = txt[["candidate_id", "profile_doc", "career_doc"]]
    meta = meta[["candidate_id", "current_title"]]

    df = feat.merge(txt, on="candidate_id", how="left").merge(meta, on="candidate_id", how="left")

    rows = []
    for _, r in df.iterrows():
        skills = []  # phase-5 table doesn't need the raw skills list here
        # use features-derived hints; keep the check deterministic and cheap
        advanced = int(r.get("advanced_skill_count", 0) or 0)
        expert = int(r.get("expert_skill_count", 0) or 0)
        yoe = float(r.get("yoe", 0) or 0)
        career_years = float(r.get("career_years_from_history", 0) or 0)
        current_title = r.get("current_title", "")
        summary_text = r.get("profile_doc", "")
        career_text = r.get("career_doc", "")

        # reconstruct a minimal skill proxy from the already-computed features
        # (the penalty logic only needs counts + text mismatch here)
        proxy_skills = [{"proficiency": "advanced"}] * advanced + [{"proficiency": "expert"}] * expert

        out = compute_consistency_score(
            yoe=yoe,
            career_years=career_years,
            skills=proxy_skills,
            current_title=current_title,
            summary_text=summary_text,
            career_text=career_text,
            advanced_skill_count=advanced,
            expert_skill_count=expert,
        )
        rows.append({
            "candidate_id": r["candidate_id"],
            **out,
        })

    cons = pd.DataFrame(rows)

    # Remove old Phase-5 consistency outputs if present
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

    cons.to_parquet(args.out, index=False)

    merged = feat.merge(
        cons,
        on="candidate_id",
        how="left",
    )

    merged.to_parquet(args.merged_out, index=False)

    print(f"Phase 6 complete. Saved: {args.out}")
    print(f"Saved merged features: {args.merged_out}")

    print(
        merged[
            [
                "consistency_score",
                "honeypot_penalty",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()
