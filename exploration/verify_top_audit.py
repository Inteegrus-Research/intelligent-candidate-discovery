#!/usr/bin/env python3
"""
Pre-Phase-8 audit for the Redrob pipeline.

What it does:
1) Verifies the Phase 7 output is sane.
2) Builds a manual-review table for the top 50 / top 100.
3) Prints bucket summaries for Top 10 / 25 / 50 / 100.
4) Flags suspicious cases so you can inspect ranking mistakes before reasoning generation.

Inputs expected:
- data/artifacts/top1000_candidates.parquet
- data/artifacts/scored_candidates.parquet
- data/raw/candidates.jsonl  (or candidates.jsonl.gz)

Outputs:
- data/artifacts/top50_audit.parquet
- data/artifacts/top50_audit.csv
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("data/artifacts")
RAW_JSONL = Path("data/raw/candidates.jsonl")
RAW_GZ = Path("data/raw/candidates.jsonl.gz")
TOP1000_PATH = BASE / "top1000_candidates.parquet"
SCORED_PATH = BASE / "scored_candidates.parquet"
AUDIT_PARQUET = BASE / "top50_audit.parquet"
AUDIT_CSV = BASE / "top50_audit.csv"


def load_raw_candidates() -> dict:
    path = RAW_GZ if RAW_GZ.exists() else RAW_JSONL
    if not path.exists():
        raise FileNotFoundError("Could not find data/raw/candidates.jsonl or candidates.jsonl.gz")

    opener = gzip.open if str(path).endswith(".gz") else open
    candidates = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            candidates[c["candidate_id"]] = c
    return candidates


def clean_text(x) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = " ".join(x.split())
    return x


def preview_text(text: str, n: int = 260) -> str:
    t = clean_text(text)
    return t[:n] + ("..." if len(t) > n else "")


def top_skills_str(candidate: dict, n: int = 6) -> str:
    skills = candidate.get("skills", []) or []
    parts = []
    for s in skills[:n]:
        name = clean_text(s.get("name", ""))
        prof = clean_text(s.get("proficiency", ""))
        end = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)
        parts.append(f"{name}({prof},e={end},d={dur})")
    return " | ".join(parts)


def career_preview(candidate: dict, n_roles: int = 2) -> str:
    roles = candidate.get("career_history", []) or []
    out = []
    for r in roles[:n_roles]:
        title = clean_text(r.get("title", ""))
        company = clean_text(r.get("company", ""))
        industry = clean_text(r.get("industry", ""))
        dur = r.get("duration_months", "?")
        out.append(f"{title} @ {company} [{industry}, {dur}mo]")
    return " || ".join(out)


def verify_top1000(top: pd.DataFrame) -> None:
    required = [
        "candidate_id",
        "retrieval_score",
        "final_score",
        "product_company_ratio",
        "honeypot_penalty",
        "consistency_score",
    ]
    missing = [c for c in required if c not in top.columns]
    if missing:
        raise ValueError(f"Top1000 missing required columns: {missing}")

    assert len(top) == 1000, f"Expected 1000 rows in top1000, found {len(top)}"
    assert top["candidate_id"].nunique() == 1000, "Duplicate candidate_id found in top1000"
    assert top["candidate_id"].duplicated().sum() == 0, "Duplicate candidate_id found in top1000"

    for c in ["retrieval_score", "final_score", "product_company_ratio", "honeypot_penalty", "consistency_score"]:
        assert top[c].isna().sum() == 0, f"NaNs found in {c}"

    # Monotonicity by final score after sorting
    top_sorted = top.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    fs = top_sorted["final_score"].to_numpy()
    if not all(fs[i] >= fs[i + 1] for i in range(len(fs) - 1)):
        raise AssertionError("final_score is not monotonic non-increasing after sorting")

    # Basic range checks
    assert float(top["product_company_ratio"].min()) >= 0.0
    assert float(top["product_company_ratio"].max()) <= 1.0
    assert float(top["honeypot_penalty"].min()) >= 0.0
    assert float(top["honeypot_penalty"].max()) <= 1.0
    assert float(top["consistency_score"].min()) >= 0.0
    assert float(top["consistency_score"].max()) <= 1.0

    print("Top1000 structural checks passed.")
    print(f"Top1000 final_score range: {top['final_score'].min():.6f} .. {top['final_score'].max():.6f}")
    print(f"Top1000 retrieval_score range: {top['retrieval_score'].min():.6f} .. {top['retrieval_score'].max():.6f}")
    print(f"Top1000 product_company_ratio mean: {top['product_company_ratio'].mean():.6f}")
    print(f"Top1000 honeypot_penalty mean: {top['honeypot_penalty'].mean():.6f}")
    print(f"Top1000 consistency_score mean: {top['consistency_score'].mean():.6f}")


def build_audit(top: pd.DataFrame, raw_map: dict) -> pd.DataFrame:
    top = top.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    rows = []
    for idx, r in top.head(100).iterrows():
        cid = r["candidate_id"]
        c = raw_map.get(cid, {})
        p = c.get("profile", {})
        sig = c.get("redrob_signals", {})
        skills = c.get("skills", [])
        career = c.get("career_history", [])

        yoe = p.get("years_of_experience")
        current_title = clean_text(p.get("current_title", ""))
        current_company = clean_text(p.get("current_company", ""))
        current_industry = clean_text(p.get("current_industry", ""))
        location = clean_text(p.get("location", ""))
        summary = p.get("summary", "")
        headline = p.get("headline", "")

        top_skill_names = [clean_text(s.get("name", "")) for s in (skills or [])[:8]]
        career_prev = career_preview(c, n_roles=3)

        rows.append({
            "rank": idx + 1,
            "candidate_id": cid,
            "final_score": float(r.get("final_score", np.nan)),
            "retrieval_score": float(r.get("retrieval_score", np.nan)),
            "current_title": current_title,
            "current_company": current_company,
            "current_industry": current_industry,
            "location": location,
            "years_of_experience": float(yoe) if yoe is not None else np.nan,
            "notice_period_days": sig.get("notice_period_days"),
            "open_to_work": sig.get("open_to_work_flag"),
            "response_rate": sig.get("recruiter_response_rate"),
            "interview_completion_rate": sig.get("interview_completion_rate"),
            "days_active": r.get("days_active", np.nan),
            "product_company_ratio": float(r.get("product_company_ratio", np.nan)),
            "consistency_score": float(r.get("consistency_score", np.nan)),
            "honeypot_penalty": float(r.get("honeypot_penalty", np.nan)),
            "skill_overlap": r.get("skill_overlap", np.nan),
            "search_experience_count": r.get("search_experience_count", np.nan),
            "career_evidence_score": r.get("career_evidence_score", np.nan),
            "profile_consistency_score": r.get("profile_consistency_score", np.nan),
            "github_activity": r.get("github_activity", np.nan),
            "profile_completeness": r.get("profile_completeness", np.nan),
            "connection_count": r.get("connection_count", np.nan),
            "endorsements_received": r.get("endorsements_received", np.nan),
            "headline": headline,
            "summary_preview": preview_text(summary, 280),
            "top_skills": " | ".join(top_skill_names),
            "career_history_preview": career_prev,
            "audit_flags": build_flags(r, p, sig, skills, career),
        })

    audit = pd.DataFrame(rows)
    return audit


def build_flags(r: pd.Series, p: dict, sig: dict, skills: list, career: list) -> str:
    flags = []

    yoe = p.get("years_of_experience")
    if yoe is not None and (yoe < 5 or yoe > 9):
        flags.append("yoe_outside_ideal_band")

    if float(r.get("honeypot_penalty", 0.0) or 0.0) >= 0.20:
        flags.append("high_honeypot_penalty")

    if float(r.get("product_company_ratio", 0.0) or 0.0) <= 0.10:
        flags.append("low_product_company_ratio")

    if float(r.get("consistency_score", 1.0) or 1.0) < 0.65:
        flags.append("low_consistency")

    if not bool(sig.get("open_to_work_flag")):
        flags.append("not_open_to_work")

    if int(sig.get("notice_period_days", 999) or 999) > 90:
        flags.append("long_notice")

    if float(sig.get("recruiter_response_rate", 0.0) or 0.0) < 0.5:
        flags.append("low_response_rate")

    if float(sig.get("interview_completion_rate", 0.0) or 0.0) < 0.5:
        flags.append("low_interview_completion")

    # Very rough semantic mismatch flag for human inspection only
    summary = clean_text(p.get("summary", ""))
    career_text = " ".join(clean_text(r.get("description", "")) for r in (career or []))
    if any(k in summary for k in ["marketing", "seo", "content", "brand"]) and any(
        k in career_text for k in ["mechanical", "cad", "solidworks", "ansys"]
    ):
        flags.append("summary_career_mismatch")

    if not flags:
        flags.append("clean")

    return ",".join(flags)


def bucket_summary(audit: pd.DataFrame, n: int) -> dict:
    d = audit.head(n)
    return {
        "n": n,
        "mean_final_score": float(d["final_score"].mean()),
        "mean_retrieval_score": float(d["retrieval_score"].mean()),
        "mean_consistency_score": float(d["consistency_score"].mean()),
        "mean_honeypot_penalty": float(d["honeypot_penalty"].mean()),
        "mean_product_company_ratio": float(d["product_company_ratio"].mean()),
        "mean_yoe": float(d["years_of_experience"].mean()),
        "mean_search_experience_count": float(d["search_experience_count"].fillna(0).mean()),
        "mean_skill_overlap": float(d["skill_overlap"].fillna(0).mean()),
        "share_flagged": float((audit.head(n)["audit_flags"] != "clean").mean()),
        "share_high_honeypot": float((d["honeypot_penalty"] >= 0.20).mean()),
        "share_low_product_company": float((d["product_company_ratio"] <= 0.10).mean()),
        "share_yoe_ideal_band": float(d["years_of_experience"].between(5, 9).mean()),
    }


def print_bucket_stats(audit: pd.DataFrame) -> None:
    for n in [10, 25, 50, 100]:
        s = bucket_summary(audit, n)
        print(f"\n=== TOP {n} ===")
        for k, v in s.items():
            if k == "n":
                continue
            print(f"{k}: {v:.6f}")


def top_candidate_warnings(audit: pd.DataFrame) -> None:
    print("\n=== Top 20 candidates for manual review ===")
    cols = [
        "rank",
        "candidate_id",
        "final_score",
        "retrieval_score",
        "years_of_experience",
        "current_title",
        "current_company",
        "product_company_ratio",
        "consistency_score",
        "honeypot_penalty",
        "notice_period_days",
        "open_to_work",
        "response_rate",
        "interview_completion_rate",
        "audit_flags",
    ]
    print(audit[cols].head(20).to_string(index=False))

    print("\n=== Candidates with warning flags in top 50 ===")
    flagged = audit.head(50)[audit.head(50)["audit_flags"] != "clean"]
    if len(flagged) == 0:
        print("No warning flags in top 50.")
    else:
        print(flagged[cols].to_string(index=False))


def main():
    for p in [TOP1000_PATH, SCORED_PATH]:
        if not p.exists():
            raise FileNotFoundError(str(p))

    top = pd.read_parquet(TOP1000_PATH)
    scored = pd.read_parquet(SCORED_PATH)
    raw_map = load_raw_candidates()

    print("=== Structural checks ===")
    verify_top1000(top)

    # Cross-check scored file if it carries final_score for the same ids
    if "final_score" in scored.columns:
        top_ids = set(top["candidate_id"].astype(str))
        scored_ids = set(scored.loc[scored["final_score"].notna(), "candidate_id"].astype(str))
        if not top_ids.issubset(scored_ids):
            print("WARNING: top1000 ids not fully present in scored file final_score coverage.")

    audit = build_audit(top, raw_map)
    audit = audit.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    audit.to_parquet(AUDIT_PARQUET, index=False)
    audit.to_csv(AUDIT_CSV, index=False)

    print("\nSaved audit files:")
    print(AUDIT_PARQUET)
    print(AUDIT_CSV)

    print_bucket_stats(audit)
    top_candidate_warnings(audit)

    print("\n=== Manual-review readiness ===")
    print("Top 10 candidates with strongest fit signals should look hireable.")
    print("Top 25 should be defensible in a judge interview.")
    print("Top 50 should not contain obvious junk.")
    print("If those are true, move to Phase 8.")


if __name__ == "__main__":
    main()
