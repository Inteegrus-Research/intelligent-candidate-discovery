#!/usr/bin/env python3
"""
Pre-Phase-8 audits A, B, C for the Redrob pipeline.

Audit A: Top 25 profiles for manual hireability check.
Audit B: Rank 900-1000 profiles to confirm lower quality.
Audit C: Keyword frequency in top 50 profiles.

Inputs:
- data/artifacts/top1000_candidates.parquet
- data/raw/candidates.jsonl  (or .jsonl.gz)
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

# Constants
BASE = Path("data/artifacts")
RAW_JSONL = Path("data/raw/candidates.jsonl")
RAW_GZ = Path("data/raw/candidates.jsonl.gz")
TOP1000_PATH = BASE / "top1000_candidates.parquet"

# Keywords for Audit C
SEARCH_RE = re.compile(
    r"\b(search|retrieval|ranking|recommendation|vector search|rag|llm retrieval|information retrieval)\b",
    re.IGNORECASE,
)

def load_raw_candidates() -> dict:
    path = RAW_GZ if RAW_GZ.exists() else RAW_JSONL
    if not path.exists():
        raise FileNotFoundError("Missing raw candidates file")
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

def top_skills(candidate: dict, n: int = 5) -> str:
    skills = candidate.get("skills", []) or []
    names = [s.get("name", "") for s in skills[:n]]
    return ", ".join(names)

def career_brief(candidate: dict, n_roles: int = 2) -> str:
    roles = candidate.get("career_history", []) or []
    parts = []
    for r in roles[:n_roles]:
        title = r.get("title", "")
        company = r.get("company", "")
        industry = r.get("industry", "")
        dur = r.get("duration_months", "?")
        parts.append(f"{title} @ {company} ({industry}, {dur}mo)")
    return " || ".join(parts)

def print_compact_profile(rank, cid, candidate, features):
    p = candidate["profile"]
    sig = candidate["redrob_signals"]
    print(f"#{rank} {cid} | {p.get('current_title')} | YOE:{p.get('years_of_experience')} | "
          f"Location:{p.get('location')} | ProductRatio:{features.get('product_company_ratio',0):.2f} | "
          f"Consistency:{features.get('consistency_score',0):.2f} | Honeypot:{features.get('honeypot_penalty',0):.2f}")
    print(f"  Notice:{sig.get('notice_period_days')}d | OpenToWork:{sig.get('open_to_work_flag')} | "
          f"RespRate:{sig.get('recruiter_response_rate')} | InterviewComp:{sig.get('interview_completion_rate')}")
    print(f"  Skills: {top_skills(candidate, 5)}")
    print(f"  Career: {career_brief(candidate, 2)}")
    print(f"  Summary snippet: {p.get('summary','')[:200]}")
    print()

def main():
    if not TOP1000_PATH.exists():
        raise FileNotFoundError("top1000_candidates.parquet missing. Run Phase 7 first.")
    
    top = pd.read_parquet(TOP1000_PATH)
    # Ensure it's sorted by final_score descending
    top = top.sort_values(
        by=["final_score", "retrieval_score", "candidate_id"],
        ascending=[False, False, True],
        kind="mergesort"
    ).reset_index(drop=True)
    
    raw_map = load_raw_candidates()
    
    # ----- AUDIT A: Top 25 -----
    print("=" * 80)
    print("AUDIT A: TOP 25 PROFILES")
    print("=" * 80)
    top25 = top.head(25)
    for idx, row in top25.iterrows():
        cid = row["candidate_id"]
        candidate = raw_map.get(cid)
        if candidate is None:
            print(f"Missing raw data for {cid}")
            continue
        print_compact_profile(idx + 1, cid, candidate, row)
    
    # ----- AUDIT B: Ranks 900-1000 -----
    print("\n" + "=" * 80)
    print("AUDIT B: RANKS 900-1000 (bottom of top1000)")
    print("=" * 80)
    bottom100 = top.iloc[899:1000].copy()
    bottom100 = bottom100.reset_index(drop=True)
    # print first 10 of those to get a feel
    for idx, row in bottom100.head(10).iterrows():
        cid = row["candidate_id"]
        candidate = raw_map.get(cid)
        if candidate is None:
            continue
        print_compact_profile(900 + idx + 1, cid, candidate, row)
    print("...")
    # also show last 5
    for idx, row in bottom100.tail(5).iterrows():
        cid = row["candidate_id"]
        candidate = raw_map.get(cid)
        if candidate is None:
            continue
        print_compact_profile(900 + idx + 1, cid, candidate, row)
    
    # ----- AUDIT C: Keywords in Top 50 -----
    print("=" * 80)
    print("AUDIT C: SEARCH/RETRIEVAL KEYWORD FREQUENCY IN TOP 50")
    print("=" * 80)
    top50 = top.head(50)
    keyword_counts = {}
    total = 0
    for _, row in top50.iterrows():
        cid = row["candidate_id"]
        candidate = raw_map.get(cid)
        if candidate is None:
            continue
        total += 1
        # Build a text blob from headline, summary, skills, career descriptions
        p = candidate["profile"]
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        text = f"{p.get('headline','')} {p.get('summary','')} " \
               f"{' '.join(s.get('name','') for s in skills)} " \
               f"{' '.join(r.get('description','') for r in career)}"
        matches = SEARCH_RE.findall(text)
        for match in matches:
            keyword = match.lower()
            if keyword == "information retrieval":
                keyword = "information retrieval"
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    print(f"Total top-50 candidates with raw data: {total}")
    for keyword, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
        print(f"  {keyword}: {count} ({count/total*100:.0f}%)")
    
    # Overall summary
    any_match = sum(1 for _, row in top50.iterrows() if row["candidate_id"] in raw_map and SEARCH_RE.search(
        f"{raw_map[row['candidate_id']]['profile'].get('headline','')} {raw_map[row['candidate_id']]['profile'].get('summary','')} "
        f"{' '.join(s.get('name','') for s in raw_map[row['candidate_id']].get('skills',[]))} "
        f"{' '.join(r.get('description','') for r in raw_map[row['candidate_id']].get('career_history',[]))}"
    ))
    print(f"\nCandidates with at least one search/retrieval keyword: {any_match}/{total} ({any_match/total*100:.0f}%)")

if __name__ == "__main__":
    main()
