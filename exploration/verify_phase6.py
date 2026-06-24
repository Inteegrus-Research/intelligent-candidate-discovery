#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

p = Path("data/artifacts/candidate_features_phase6.parquet")
df = pd.read_parquet(p)

required = [
    "consistency_score",
    "honeypot_penalty",
    "yoe_alignment_score",
    "domain_alignment_score",
    "skill_evidence_score",
]

print("rows:", len(df))
print("unique ids:", df["candidate_id"].nunique())
print("dup ids:", df["candidate_id"].duplicated().sum())
print("missing required cols:", [c for c in required if c not in df.columns])

print("\nNaN rates:")
print(df[required].isna().mean().sort_values(ascending=False))

print("\nRanges:")
for c in required:
    print(c, float(df[c].min()), float(df[c].max()))

print("\nTop honeypots:")
print(
    df.sort_values("honeypot_penalty", ascending=False)[
        ["candidate_id", "honeypot_penalty", "consistency_score", "yoe_alignment_score", "domain_alignment_score"]
    ].head(10).to_string(index=False)
)

print("\nMemory MB:", round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
