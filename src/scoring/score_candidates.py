#!/usr/bin/env python3
"""
Phase 7: score and rerank candidates.

Inputs:
- data/artifacts/candidate_features_phase6.parquet

Outputs:
- data/artifacts/top1000_candidates.parquet
- data/artifacts/scored_candidates.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEMANTIC_W = {
    "profile_similarity": 0.10,
    "skills_similarity": 0.10,
    "career_similarity": 0.10,
    "full_similarity": 0.10,
    "intent1_similarity": 0.10,
    "intent2_similarity": 0.10,
    "intent3_similarity": 0.10,
    "intent4_similarity": 0.10,
    "intent5_similarity": 0.10,
    "semantic_alignment_score": 0.15,
    "intent_weighted_full_similarity": 0.05,
}

CAREER_W = {
    "has_search_experience": 0.25,
    "search_experience_count": 0.15,
    "career_evidence_score": 0.15,
    "build_ownership_count": 0.10,
    "product_company_ratio": 0.35,
}

SKILL_W = {
    "skill_overlap": 0.18,
    "weighted_skill_overlap": 0.28,
    "advanced_skill_count": 0.12,
    "expert_skill_count": 0.12,
    "assessment_score": 0.20,
    "skill_coverage_ratio": 0.10,
}

BEHAVIORAL_W = {
    "response_rate": 0.20,
    "interview_completion": 0.15,
    "days_active": 0.10,
    "profile_views": 0.12,
    "saved_by_recruiters": 0.12,
    "search_appearance": 0.08,
    "open_to_work": 0.10,
    "response_speed_score": 0.08,
    "activity_recency_score": 0.05,
}

AVAILABILITY_W = {
    "notice_score": 0.25,
    "verified_email": 0.12,
    "verified_phone": 0.12,
    "linkedin_connected": 0.08,
    "offer_acceptance": 0.08,
    "location_match": 0.18,
    "work_mode_match": 0.10,
    "willing_to_relocate": 0.07,
}

QUALITY_W = {
    "profile_completeness": 0.18,
    "connection_count": 0.12,
    "endorsements_received": 0.12,
    "github_activity": 0.12,
    "profile_consistency_score": 0.20,
    "consistency_score": 0.26,
}


def minmax_01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mn) / (mx - mn)


def robust_clip(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    if hi == lo:
        return s
    return s.clip(lo, hi)


def normalize_group(df: pd.DataFrame, cols: list[str], invert: set[str] | None = None) -> pd.DataFrame:
    invert = invert or set()
    out = {}
    for c in cols:
        if c not in df.columns:
            out[c] = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            continue

        x = df[c].copy()

        if c in {
            "connection_count",
            "endorsements_received",
            "profile_views",
            "saved_by_recruiters",
            "search_appearance",
            "skill_overlap",
            "weighted_skill_overlap",
            "search_experience_count",
            "build_ownership_count",
        }:
            x = robust_clip(x)

        x = minmax_01(x)

        if c in invert:
            x = 1.0 - x

        out[c] = x

    return pd.DataFrame(out, index=df.index)


def weighted_sum(df: pd.DataFrame, weights: dict[str, float], invert: set[str] | None = None) -> pd.Series:
    norm = normalize_group(df, list(weights.keys()), invert=invert)
    score = np.zeros(len(df), dtype=float)
    for c, w in weights.items():
        score += norm[c].to_numpy(dtype=float) * float(w)
    return pd.Series(score, index=df.index, dtype=float)


def fallback_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "product_company_ratio" not in df.columns:
        if "service_company_ratio" in df.columns:
            df["product_company_ratio"] = 1.0 - df["service_company_ratio"].fillna(0.0).astype(float)
        else:
            df["product_company_ratio"] = 0.0

    if "activity_recency_score" not in df.columns and "days_active" in df.columns:
        df["activity_recency_score"] = np.exp(-df["days_active"].fillna(9999).astype(float) / 90.0)

    if "response_speed_score" not in df.columns and "avg_response_time_hours" in df.columns:
        df["response_speed_score"] = np.exp(-df["avg_response_time_hours"].fillna(9999).astype(float) / 72.0)

    if "days_active" not in df.columns and "days_since_active" in df.columns:
        df["days_active"] = df["days_since_active"]

    if "profile_views" not in df.columns and "views_received_30d" in df.columns:
        df["profile_views"] = df["views_received_30d"]

    if "saved_by_recruiters" not in df.columns and "saved_by_recruiters_30d" in df.columns:
        df["saved_by_recruiters"] = df["saved_by_recruiters_30d"]

    if "search_appearance" not in df.columns and "search_appearance_30d" in df.columns:
        df["search_appearance"] = df["search_appearance_30d"]

    return df


def compute_retrieval_score(df: pd.DataFrame) -> pd.Series:
    cols = [
        "full_similarity",
        "intent_weighted_full_similarity",
        "semantic_alignment_score",
        "career_similarity",
        "skills_similarity",
        "profile_similarity",
    ]
    x = df[cols].copy()
    for c in cols:
        x[c] = robust_clip(x[c])
        x[c] = minmax_01(x[c])

    score = (
        0.28 * x["full_similarity"]
        + 0.22 * x["intent_weighted_full_similarity"]
        + 0.18 * x["semantic_alignment_score"]
        + 0.14 * x["career_similarity"]
        + 0.10 * x["skills_similarity"]
        + 0.08 * x["profile_similarity"]
    )
    return score.astype(float)


def compute_final_score(df: pd.DataFrame) -> pd.Series:
    semantic = weighted_sum(df, SEMANTIC_W)
    career = weighted_sum(df, CAREER_W)
    skill = weighted_sum(df, SKILL_W)
    behavioral = weighted_sum(df, BEHAVIORAL_W, invert={"days_active"})
    availability = weighted_sum(df, AVAILABILITY_W)
    quality = weighted_sum(df, QUALITY_W)

    base = (
        0.40 * semantic
        + 0.25 * career
        + 0.15 * skill
        + 0.10 * behavioral
        + 0.05 * availability
        + 0.05 * quality
    )

    honeypot = df["honeypot_penalty"].fillna(0.0).astype(float).clip(0.0, 1.0)
    final = base * (1.0 - honeypot)
    return pd.Series(final, index=df.index, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/artifacts/candidate_features_phase6.parquet")
    ap.add_argument("--out-top1000", default="data/artifacts/top1000_candidates.parquet")
    ap.add_argument("--out-scored", default="data/artifacts/scored_candidates.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.features).copy()
    df = fallback_columns(df)

    required = [
        "candidate_id",
        "full_similarity",
        "intent_weighted_full_similarity",
        "semantic_alignment_score",
        "career_similarity",
        "skills_similarity",
        "profile_similarity",
        "honeypot_penalty",
        "consistency_score",
        "product_company_ratio",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Phase 7: {missing}")

    # Retrieval stage
    df["retrieval_score"] = compute_retrieval_score(df)

    n = len(df)
    top_n = min(1000, n)
    kth = min(top_n - 1, n - 1)
    top_idx = np.argpartition(-df["retrieval_score"].to_numpy(), kth)[:top_n]

    top_df = df.iloc[top_idx].copy()
    top_df = top_df.sort_values(
        by=["retrieval_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # Rerank only the retrieved pool
    top_df["final_score"] = compute_final_score(top_df)

    top_df = top_df.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # Save top-1000 and full scored pool
    top_df.to_parquet(args.out_top1000, index=False)

    full_scored = df.copy()
    full_scored["final_score"] = np.nan
    final_map = pd.Series(top_df["final_score"].to_numpy(), index=top_df["candidate_id"].astype(str))
    full_scored["final_score"] = full_scored["candidate_id"].astype(str).map(final_map)
    full_scored.to_parquet(args.out_scored, index=False)

    print("Phase 7 complete.")
    print(f"Saved: {args.out_top1000}")
    print(f"Saved: {args.out_scored}")
    print(
        top_df[
            ["candidate_id", "retrieval_score", "final_score", "product_company_ratio", "honeypot_penalty", "consistency_score"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
