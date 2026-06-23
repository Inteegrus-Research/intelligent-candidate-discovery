#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

path = Path("data/artifacts/candidate_features_phase5.parquet")
df = pd.read_parquet(path)

required = [
    "profile_similarity", "skills_similarity", "career_similarity", "full_similarity",
    "intent1_similarity", "intent2_similarity", "intent3_similarity", "intent4_similarity", "intent5_similarity",
    "skill_overlap", "weighted_skill_overlap", "advanced_skill_count", "expert_skill_count", "assessment_score",
    "has_search_experience", "search_experience_count",
    "product_company_ratio",
    "response_rate", "interview_completion", "days_active", "profile_views",
    "saved_by_recruiters", "search_appearance", "open_to_work",
    "notice_score", "verified_email", "verified_phone", "linkedin_connected", "offer_acceptance",
    "profile_completeness", "connection_count", "endorsements_received", "github_activity",
    "profile_consistency_score", "honeypot_penalty",
]

print("rows:", len(df))
print("unique ids:", df["candidate_id"].nunique())
print("dup ids:", df["candidate_id"].duplicated().sum())

missing = [c for c in required if c not in df.columns]
print("missing required cols:", missing)

check_cols = [c for c in required if c in df.columns]
print("\nNaN rate top 20:")
print(df[check_cols].isna().mean().sort_values(ascending=False).head(20))

print("\nRanges:")
for c in ["profile_similarity", "skills_similarity", "career_similarity", "full_similarity",
          "intent1_similarity", "intent2_similarity", "intent3_similarity", "intent4_similarity", "intent5_similarity"]:
    if c in df.columns:
        print(c, float(df[c].min()), float(df[c].max()))
for c in ["product_company_ratio", "notice_score", "profile_consistency_score", "honeypot_penalty"]:
    if c in df.columns:
        print(c, float(df[c].min()), float(df[c].max()))

print("\nMissing flags:")
for c in ["github_activity_missing", "offer_acceptance_missing"]:
    if c in df.columns:
        print(c, int(df[c].sum()))

print("\nMemory MB:", round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
print("\nSample:")
print(df[["candidate_id", "skill_overlap", "product_company_ratio", "profile_consistency_score", "honeypot_penalty"]].head(5).to_string(index=False))
