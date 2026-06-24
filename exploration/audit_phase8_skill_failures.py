#!/usr/bin/env python3

from pathlib import Path
import gzip
import json
import re
import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+\-/]*", re.I)

def norm(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x).strip()).lower()

def load_raw(path):
    p = Path(path)

    if not p.exists():
        gz = Path(str(p) + ".gz")
        if gz.exists():
            p = gz
        else:
            raise FileNotFoundError(path)

    opener = gzip.open if str(p).endswith(".gz") else open

    out = {}

    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            out[c["candidate_id"]] = c

    return out


raw = load_raw("data/raw/candidates.jsonl")
sub = pd.read_csv("submissions/team_name.csv")

failed = []

for _, row in sub.iterrows():

    cid = row["candidate_id"]
    reasoning = norm(row["reasoning"])

    candidate = raw[cid]

    skills = candidate.get("skills", [])

    skill_names = []

    for s in skills:
        n = norm(s.get("name", ""))

        if n:
            skill_names.append(n)

    exact_matches = []

    for s in skill_names:
        if s in reasoning:
            exact_matches.append(s)

    if len(exact_matches) == 0:
        failed.append(
            {
                "candidate_id": cid,
                "reasoning": row["reasoning"],
                "skills": skill_names[:15],
            }
        )

print()
print("=" * 80)
print("FAILED SKILL MATCHES")
print("=" * 80)
print()

print("count:", len(failed))
print()

for item in failed[:30]:

    print("-" * 80)
    print(item["candidate_id"])
    print()

    print("RAW SKILLS:")
    print(item["skills"])

    print()
    print("REASONING:")
    print(item["reasoning"])
    print()

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(
    f"Failed candidates = {len(failed)} / {len(sub)} "
    f"({100*len(failed)/len(sub):.1f}%)"
)
