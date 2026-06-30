#!/usr/bin/env python3
"""
Pipeline Performance Profiler — India Runs Hackathon

Measures:
- Full online pipeline runtime (loading → ranking → CSV)
- Individual stage timings
- Peak memory usage during ranking
- Throughput (candidates ranked per second)
- Scalability (reduction from 100k to top 100)
- Compute environment: CPU‑only, no GPU, RAM < 16 GB

Uses actual scoring function from src/scoring/rank_candidates.py
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tracemalloc  # built‑in, for memory tracking

# Fix import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from scoring.rank_candidates import compute_final_score
except ImportError:
    print("Cannot import scoring.rank_candidates.compute_final_score. Aborting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("data/artifacts")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

TOP1000_FILE = ARTIFACTS_DIR / "retrieval_top1000.parquet"
EMBEDDINGS_FILE = ARTIFACTS_DIR / "candidate_embeddings.npy"
JD_EMB_FILE = ARTIFACTS_DIR / "jd_intent_embeddings.npy"
IDS_FILE = ARTIFACTS_DIR / "candidate_ids.npy"
SUBMISSION_OUT = Path("submissions")  # we won't overwrite, just time writing

# ---------------------------------------------------------------------------
# Profiling helper
# ---------------------------------------------------------------------------
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

# ---------------------------------------------------------------------------
# Main profiling
# ---------------------------------------------------------------------------
def profile():
    results = {"runtime": {}, "resources": {}, "scalability": {}, "environment": {}}
    
    print("Profiling pipeline...")
    tracemalloc.start()
    
    # 1. JD parsing & intent embedding – these are precomputed, so we measure loading
    with Timer("JD intent embedding loading") as t:
        jd_emb = np.load(JD_EMB_FILE).astype(np.float32)
    results["runtime"]["jd_embedding_load"] = round(t.elapsed, 6)

    # 2. Candidate embeddings loading
    with Timer("Candidate embeddings loading") as t:
        cand_emb = np.load(EMBEDDINGS_FILE, mmap_mode="r")  # memory‑mapped
    results["runtime"]["candidate_embeddings_load"] = round(t.elapsed, 6)

    # 3. Candidate IDs loading
    with Timer("Candidate IDs loading") as t:
        ids = np.load(IDS_FILE, allow_pickle=True)
    results["runtime"]["candidate_ids_load"] = round(t.elapsed, 6)

    # 4. Top‑1000 features loading (the retrieval pool)
    with Timer("Top‑1000 features loading") as t:
        top_df = pd.read_parquet(TOP1000_FILE)
    results["runtime"]["top1000_features_load"] = round(t.elapsed, 6)

    # 5. Retrieval score computation (simulate using dot product)
    # We already have retrieval scores in top_df, but for completeness we measure what a retrieval would take.
    # We'll time the dot product of candidate full‑doc embeddings with JD intents.
    full_doc_emb = cand_emb[:, 3, :]  # memory‑mapped, actual loading happens on access
    with Timer("Semantic retrieval (cosine similarity)") as t:
        # Compute cosine similarities: N x D dot D → N
        retrieval_scores = np.dot(full_doc_emb.astype(np.float32), jd_emb.mean(axis=0))
    results["runtime"]["semantic_retrieval"] = round(t.elapsed, 6)

    # 6. Feature engineering & final scoring using the actual function
    # We measure the time to compute final scores for the entire top‑1000.
    with Timer("Hybrid ranking (final score computation)") as t:
        final_scores = compute_final_score(top_df)
    results["runtime"]["hybrid_ranking"] = round(t.elapsed, 6)

    # 7. Submission generation (top‑100 selection + CSV write)
    with Timer("Submission generation (top‑100 + CSV)") as t:
        # Sort and take top 100
        ranked = top_df.copy()
        ranked["final_score"] = final_scores
        ranked = ranked.sort_values(
            by=["final_score", "retrieval_score", "candidate_id"],
            ascending=[False, False, True],
        ).head(100)
        ranked[["candidate_id", "final_score"]].to_csv(SUBMISSION_OUT / "profile_test.csv", index=False)
    results["runtime"]["submission_generation"] = round(t.elapsed, 6)

    # 8. Total online pipeline runtime (sum of all measured stages after embeddings are loaded)
    total_online = sum(
        results["runtime"].get(k, 0) for k in [
            "jd_embedding_load",
            "candidate_embeddings_load",
            "candidate_ids_load",
            "top1000_features_load",
            "semantic_retrieval",
            "hybrid_ranking",
            "submission_generation",
        ]
    )
    results["runtime"]["total_online_pipeline"] = round(total_online, 4)

    # Throughput
    throughput = len(top_df) / results["runtime"]["hybrid_ranking"] if results["runtime"]["hybrid_ranking"] > 0 else 0
    results["resources"]["throughput_candidates_per_sec"] = round(throughput, 1)

    # Scalability
    results["scalability"] = {
        "total_candidates": len(ids),
        "retrieved_candidates": len(top_df),
        "final_shortlist": 100,
        "reduction_pct": round((1 - 100/len(ids)) * 100, 2),
    }

    # Memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["resources"]["peak_memory_mb"] = round(peak / 1024 / 1024, 2)
    results["resources"]["current_memory_mb"] = round(current / 1024 / 1024, 2)

    # Environment
    results["environment"] = {
        "cpu_cores": 1,  # we run single‑threaded
        "gpu_required": False,
        "ram_limit_gb": 16,
        "disk_usage_submission_kb": round(
            (SUBMISSION_OUT / "profile_test.csv").stat().st_size / 1024, 1
        ) if (SUBMISSION_OUT / "profile_test.csv").exists() else None,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("PERFORMANCE PROFILE")
    print("=" * 60)
    for stage, sec in results["runtime"].items():
        print(f"{stage:40s} : {sec:.6f} s")
    print(f"\nThroughput (candidates/sec) : {results['resources']['throughput_candidates_per_sec']:.1f}")
    print(f"Peak RAM usage              : {results['resources']['peak_memory_mb']:.2f} MB")
    print(f"Scalability                 : {results['scalability']['reduction_pct']:.1f}% reduction")
    print(f"Submission size             : {results['environment']['disk_usage_submission_kb']} KB")
    print(f"GPU required                : {results['environment']['gpu_required']}")

    # Save to JSON
    report_path = REPORTS_DIR / "performance_profile.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nProfile saved to {report_path}")

if __name__ == "__main__":
    profile()
