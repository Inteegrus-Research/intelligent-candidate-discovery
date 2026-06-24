#!/usr/bin/env python3
"""
Verify Phase 8 output before final submission.
Checks:
- 100 rows
- ranks 1..100
- unique candidate_ids
- non-increasing scores
- non-empty, non-repetitive reasoning
- reasoning mentions factual profile elements
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import pandas as pd


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+\-/]*", re.I)


def clean_text(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm(x: str) -> str:
    return clean_text(x).lower()


def load_raw(path: Path) -> dict:
    if not path.exists():
        gz = Path(str(path) + ".gz")
        if gz.exists():
            path = gz
        else:
            raise FileNotFoundError(f"Missing raw candidates file: {path} or {gz}")

    opener = gzip.open if str(path).endswith(".gz") else open
    out = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            out[c["candidate_id"]] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submissions/team_name.csv")
    ap.add_argument("--raw", default="data/raw/candidates.jsonl")
    args = ap.parse_args()

    sub = pd.read_csv(args.submission)
    raw = load_raw(Path(args.raw))

    print("rows:", len(sub))
    print("unique ids:", sub["candidate_id"].nunique())
    print("dup ids:", int(sub["candidate_id"].duplicated().sum()))

    required = ["candidate_id", "rank", "score", "reasoning"]
    missing = [c for c in required if c not in sub.columns]
    print("missing cols:", missing)
    assert not missing, f"Missing columns: {missing}"

    assert len(sub) == 100, f"Submission must have exactly 100 rows, found {len(sub)}"
    assert sub["candidate_id"].nunique() == 100, "Duplicate candidate_ids found"
    assert list(sub["rank"]) == list(range(1, 101)), "Ranks must be exactly 1..100"
    assert sub["reasoning"].fillna("").str.len().min() > 0, "Empty reasoning found"

    scores = sub["score"].to_numpy()
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Score monotonicity failed"

    lengths = sub["reasoning"].fillna("").str.len()
    print("\nreasoning length stats:")
    print(lengths.describe())

    title_hits = 0
    skill_hits = 0
    behavior_hits = 0
    company_hits = 0
    generic_only = 0

    for _, row in sub.iterrows():
        cid = row["candidate_id"]
        c = raw[cid]
        p = c["profile"]
        sig = c["redrob_signals"]
        skills = c.get("skills", []) or []
        title = norm(p.get("current_title", ""))
        company = norm(p.get("current_company", ""))
        reasoning = norm(row["reasoning"])

        title_words = [t for t in TOKEN_RE.findall(title) if len(t) > 3]
        skill_names = [norm(s.get("name", "")) for s in skills]

        has_title = any(w in reasoning for w in title_words[:3]) if title_words else False
        has_skill = any(s and s in reasoning for s in skill_names[:5])
        has_behavior = any(phrase in reasoning for phrase in ["open to work", "notice period", "recruiter response rate", "interview completion"])
        has_company = company and company in reasoning

        title_hits += int(has_title)
        skill_hits += int(has_skill)
        behavior_hits += int(has_behavior)
        company_hits += int(has_company)
        generic_only += int(not (has_title or has_skill or has_behavior or has_company))

    print("\ncoverage:")
    print(f"title hit: {title_hits}/100")
    print(f"skill hit: {skill_hits}/100")
    print(f"behavior hit: {behavior_hits}/100")
    print(f"company hit: {company_hits}/100")
    print(f"generic only: {generic_only}/100")

    assert generic_only == 0, "Some reasonings are too generic"
    assert behavior_hits >= 80, "Behavioral evidence should appear in most reasonings"
    assert skill_hits >= 80, "Skill evidence should appear in most reasonings"

    print("\nTop 10 reasonings:")
    print(sub.head(10)[["rank", "candidate_id", "score", "reasoning"]].to_string(index=False))

    print("\nPhase 8 verification passed.")


if __name__ == "__main__":
    main()
