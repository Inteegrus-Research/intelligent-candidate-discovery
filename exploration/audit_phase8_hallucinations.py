#!/usr/bin/env python3

import gzip
import json
import pandas as pd
from pathlib import Path

ALLOWED = {
    "faiss",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "rag",
    "vector search",
    "information retrieval",
    "learning to rank",
    "sentence transformers",
    "recommendation systems",
}

def norm(x):
    return str(x).strip().lower()

def load_raw(path):
    p = Path(path)

    if not p.exists():
        p = Path(str(path) + ".gz")

    opener = gzip.open if str(p).endswith(".gz") else open

    out = {}

    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            out[c["candidate_id"]] = c

    return out


raw = load_raw("data/raw/candidates.jsonl")
sub = pd.read_csv("submissions/team_name.csv")

issues = []

for _, row in sub.iterrows():

    cid = row["candidate_id"]

    reasoning = row["reasoning"].lower()

    candidate = raw[cid]

    skill_set = {
        norm(s.get("name", ""))
        for s in candidate.get("skills", [])
    }

    for term in ALLOWED:

        if term in reasoning and term not in skill_set:
            issues.append((cid, term))

print("possible hallucinations:", len(issues))

for x in issues[:50]:
    print(x)
