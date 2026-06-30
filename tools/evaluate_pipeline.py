#!/usr/bin/env python3
"""
Pipeline Evaluation Suite — India Runs Hackathon (v2.1.2)

* Exact feature group contribution via ablation using the actual scoring function
* Ranking runtime profiling
* Comprehensive metrics, plots, and reports
* No synthetic IR metrics (Precision/Recall/NDCG)
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is in the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from scoring.rank_candidates import (
        compute_final_score,
        SEMANTIC_W,
        CAREER_W,
        SKILL_W,
        BEHAVIORAL_W,
        AVAILABILITY_W,
        QUALITY_W,
    )
except ImportError as e:
    print(f"Error importing scoring module: {e}")
    print("Make sure the module exists and contains compute_final_score and weight dictionaries.")
    sys.exit(1)

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path("data/artifacts")
REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TOP1000_FILE = ARTIFACTS_DIR / "retrieval_top1000.parquet"
FEATURES_FILE = ARTIFACTS_DIR / "candidate_features.parquet"
METADATA_FILE = ARTIFACTS_DIR / "candidate_metadata.parquet"
IDS_FILE = ARTIFACTS_DIR / "candidate_ids.npy"
SUBMISSION_FILE = Path("submissions/submission.csv")

# Feature groups for correlation heatmap
FEATURE_GROUPS = {
    "Semantic": [
        "retrieval_score", "profile_similarity", "skills_similarity",
        "career_similarity", "full_similarity", "intent_weighted_full_similarity",
        "semantic_alignment_score",
    ],
    "Quality": [
        "consistency_score", "profile_completeness", "product_company_ratio",
        "honeypot_penalty",
    ],
    "Behavioral": [
        "response_rate", "interview_completion", "days_active",
        "profile_views", "saved_by_recruiters", "search_appearance",
        "open_to_work", "activity_recency_score", "response_speed_score",
    ],
    "Availability": [
        "notice_period_days", "verified_email", "verified_phone",
        "linkedin_connected", "offer_acceptance", "location_match",
        "work_mode_match", "willing_to_relocate",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_artifact(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        return pd.DataFrame({"value": data})
    return pd.read_parquet(path, **kwargs)

def save_plot(fig, name: str):
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot: {path}")

def json_safe(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class PipelineEvaluator:
    def __init__(self):
        self.results = {}
        self.top1000 = None
        self.features = None
        self.metadata = None
        self.submission = None
        self.candidate_ids = None

    def load_data(self):
        self.top1000 = load_artifact(TOP1000_FILE)
        self.features = load_artifact(FEATURES_FILE)
        self.metadata = load_artifact(METADATA_FILE)
        self.submission = load_artifact(SUBMISSION_FILE)
        ids = np.load(IDS_FILE, allow_pickle=True)
        self.candidate_ids = ids

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------
    def dataset_stats(self):
        total = len(self.candidate_ids)
        retrieved = len(self.top1000)
        final = len(self.submission)
        meta = self.metadata
        avg_exp = meta["years_of_experience"].mean() if "years_of_experience" in meta.columns else None
        median_exp = meta["years_of_experience"].median() if "years_of_experience" in meta.columns else None
        unique_titles = meta["current_title"].nunique() if "current_title" in meta.columns else None
        unique_companies = meta["current_company"].nunique() if "current_company" in meta.columns else None
        open_to_work_pct = meta["open_to_work_flag"].mean() * 100 if "open_to_work_flag" in meta.columns else None
        self.results["dataset"] = {
            "total_candidates": total,
            "retrieved_pool_size": retrieved,
            "final_shortlist_size": final,
            "avg_years_of_experience": round(avg_exp, 2) if avg_exp is not None else None,
            "median_years_of_experience": round(median_exp, 2) if median_exp is not None else None,
            "unique_job_titles": unique_titles,
            "unique_companies": unique_companies,
            "open_to_work_pct": round(open_to_work_pct, 1) if open_to_work_pct is not None else None,
        }

    def retrieval_stats(self):
        scores = self.top1000["retrieval_score"]
        self.results["retrieval"] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "p25": float(scores.quantile(0.25)),
            "median": float(scores.median()),
            "p75": float(scores.quantile(0.75)),
            "max": float(scores.max()),
        }

    def final_score_stats(self):
        scores = self.top1000["final_score"]
        self.results["final_score"] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "p25": float(scores.quantile(0.25)),
            "median": float(scores.median()),
            "p75": float(scores.quantile(0.75)),
            "max": float(scores.max()),
        }

    def ranking_quality(self):
        top = self.top1000.head(100)["final_score"].reset_index(drop=True)
        if len(top) < 100:
            return
        gaps = top.iloc[:-1].values - top.iloc[1:].values
        self.results["ranking_quality"] = {
            "top10_avg": float(top[:10].mean()),
            "top25_avg": float(top[:25].mean()),
            "top50_avg": float(top[:50].mean()),
            "top100_avg": float(top[:100].mean()),
            "drop_1_to_10": float(top[0] - top[9]),
            "drop_10_to_25": float(top[9] - top[24]) if len(top) > 24 else None,
            "drop_25_to_50": float(top[24] - top[49]) if len(top) > 49 else None,
            "drop_50_to_100": float(top[49] - top[99]) if len(top) > 99 else None,
            "avg_consecutive_gap": float(np.mean(gaps)),
            "max_consecutive_gap": float(np.max(gaps)),
            "min_consecutive_gap": float(np.min(gaps)),
        }

    def diversity(self):
        top100 = self.top1000.head(100)
        merged = top100.merge(self.metadata, on="candidate_id", how="left") if not self.metadata.empty else top100
        companies = merged["current_company"] if "current_company" in merged.columns else pd.Series()
        titles = merged["current_title"] if "current_title" in merged.columns else pd.Series()
        self.results["diversity"] = {
            "unique_companies_in_top100": int(companies.nunique()) if not companies.empty else 0,
            "unique_titles_in_top100": int(titles.nunique()) if not titles.empty else 0,
            "most_represented_company": companies.value_counts().idxmax() if not companies.empty else "N/A",
            "most_represented_company_pct": round(companies.value_counts().max() / 100 * 100, 1) if not companies.empty else 0,
        }

    def quality_stats(self):
        pool = self.top1000
        metrics = {
            "consistency_score": "consistency",
            "profile_completeness": "profile_completeness",
            "product_company_ratio": "product_ratio",
            "honeypot_penalty": "honeypot_penalty",
            "retrieval_score": "semantic_retrieval",
        }
        quality = {}
        for col, label in metrics.items():
            if col in pool.columns:
                s = pool[col]
                quality[label] = {
                    "mean": round(s.mean(), 4),
                    "median": round(s.median(), 4),
                    "min": round(s.min(), 4),
                    "max": round(s.max(), 4),
                }
        self.results["candidate_quality"] = quality

    def availability_stats(self):
        pool = self.top1000
        stats = {}
        if "notice_period_days" in pool.columns:
            notice = pool["notice_period_days"]
            stats["notice_period_days"] = {
                "mean": round(notice.mean(), 1),
                "median": round(notice.median(), 1),
                "pct_leq_30": round((notice <= 30).mean() * 100, 1),
                "pct_leq_60": round((notice <= 60).mean() * 100, 1),
                "pct_leq_90": round((notice <= 90).mean() * 100, 1),
            }
        if "open_to_work" in pool.columns:
            stats["open_to_work_pct"] = round(pool["open_to_work"].mean() * 100, 1)
        if "response_rate" in pool.columns:
            stats["avg_recruiter_response"] = round(pool["response_rate"].mean(), 3)
        if "interview_completion" in pool.columns:
            stats["avg_interview_completion"] = round(pool["interview_completion"].mean(), 3)
        self.results["availability"] = stats

    def integrity_stats(self):
        penalty = self.top1000["honeypot_penalty"]
        self.results["integrity"] = {
            "mean_penalty": round(penalty.mean(), 4),
            "max_penalty": round(penalty.max(), 4),
            "pct_with_penalty": round((penalty > 0).mean() * 100, 2),
            "pct_penalty_above_01": round((penalty > 0.1).mean() * 100, 2),
        }

    def explainability_stats(self):
        if "reasoning" not in self.submission.columns:
            self.results["explainability"] = {"status": "reasoning column missing"}
            return
        reasons = self.submission["reasoning"].dropna()
        lengths = reasons.str.len()
        words = reasons.str.split().str.len()
        self.results["explainability"] = {
            "coverage_pct": round(len(reasons) / len(self.submission) * 100, 1),
            "avg_length_chars": round(lengths.mean(), 1),
            "max_length_chars": int(lengths.max()),
            "min_length_chars": int(lengths.min()),
            "avg_words": round(words.mean(), 1),
            "contains_semantic_pct": round(reasons.str.contains("semantic|retrieval|similarity", case=False).mean() * 100, 1),
            "contains_consistency_pct": round(reasons.str.contains("consistency", case=False).mean() * 100, 1),
            "contains_availability_pct": round(reasons.str.contains("notice|open to work|availability", case=False).mean() * 100, 1),
        }

    def validation(self):
        sub = self.submission
        checks = {
            "row_count_100": len(sub) == 100,
            "unique_ids": sub["candidate_id"].nunique() == 100,
            "no_duplicates": sub["candidate_id"].duplicated().sum() == 0,
            "ranks_1_to_100": list(sub["rank"]) == list(range(1, 101)),
            "score_monotonic": all(sub["score"].iloc[i] >= sub["score"].iloc[i+1] for i in range(len(sub)-1)),
            "reasoning_present": sub["reasoning"].fillna("").str.len().min() > 0,
            "ids_in_top1000": sub["candidate_id"].isin(self.top1000["candidate_id"]).all(),
            "score_in_range_0_1": sub["score"].between(0, 1).all(),
            "no_empty_reasoning": sub["reasoning"].notna().all(),
        }
        all_pass = all(checks.values())
        self.results["validation"] = {"all_passed": all_pass, "checks": {k: v for k, v in checks.items()}}

    # -------------------------------------------------------------------
    # Feature group contribution via ablation (exact scoring function)
    # -------------------------------------------------------------------
    def feature_group_contribution(self):
        df = self.top1000.copy()
        # Baseline scores
        t0 = time.time()
        base_scores = compute_final_score(df)
        rank_time = time.time() - t0
        self.results["ranking_runtime_sec"] = round(rank_time, 3)

        # Define groups (from imported weights)
        groups = {
            "semantic": list(SEMANTIC_W.keys()),
            "career": list(CAREER_W.keys()),
            "skill": list(SKILL_W.keys()),
            "behavioral": list(BEHAVIORAL_W.keys()),
            "availability": list(AVAILABILITY_W.keys()),
            "quality": list(QUALITY_W.keys()),
        }

        contributions = {}
        for grp, cols in groups.items():
            existing = [c for c in cols if c in df.columns]
            if not existing:
                continue
            # Ablate group by setting features to their minimum
            temp_df = df.copy()
            for c in existing:
                temp_df[c] = df[c].min()
            scores_without = compute_final_score(temp_df)
            contrib = float((base_scores - scores_without).mean())
            contributions[grp] = max(0, contrib)  # non-negative

        # Normalize to sum 1
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        else:
            contributions = {k: 1.0 / len(groups) for k in groups}

        self.results["feature_group_contribution_ablation"] = contributions

        # Plot horizontal bar
        fig, ax = plt.subplots()
        sorted_items = sorted(contributions.items(), key=lambda x: -x[1])
        labels, values = zip(*sorted_items)
        ax.barh(labels, values, color="steelblue")
        ax.set_xlabel("Contribution proportion")
        ax.set_title("Feature Group Contribution (Ablation Analysis)")
        save_plot(fig, "feature_group_contribution.png")

    # -------------------------------------------------------------------
    # Additional plots
    # -------------------------------------------------------------------
    def ranking_curve_plot(self):
        top100 = self.top1000.head(100)["final_score"].reset_index(drop=True)
        fig, ax = plt.subplots()
        ax.plot(range(1, len(top100)+1), top100, marker='.', linestyle='-', color='steelblue')
        ax.set_xlabel("Rank")
        ax.set_ylabel("Final Score")
        ax.set_title("Top 100 Ranking Curve")
        ax.invert_xaxis()  # rank 1 on left
        save_plot(fig, "ranking_curve.png")

    def feature_coverage_plot(self):
        # Feature availability across the top1000
        pool = self.top1000
        coverage = {}
        for col in pool.columns:
            if col == "candidate_id":
                continue
            coverage[col] = pool[col].notna().mean() * 100
        if coverage:
            cov_df = pd.DataFrame(list(coverage.items()), columns=["feature", "coverage_pct"])
            cov_df = cov_df.sort_values("coverage_pct", ascending=False)
            cov_df.to_csv(REPORTS_DIR / "feature_availability.csv", index=False)
            # Plot top 20 features
            top_features = cov_df.head(20)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_features["feature"], top_features["coverage_pct"], color="teal")
            ax.set_xlabel("Coverage (%)")
            ax.set_title("Feature Coverage (Top 20)")
            ax.invert_yaxis()
            save_plot(fig, "feature_coverage.png")

    # -------------------------------------------------------------------
    # Candidate archetypes
    # -------------------------------------------------------------------
    def candidate_archetypes(self):
        top = self.top1000.head(100)
        best = top.iloc[0]
        median_idx = len(top) // 2
        median = top.iloc[median_idx]
        lowest = top.iloc[-1]
        archetypes = {}
        for name, row in [("best", best), ("median", median), ("lowest", lowest)]:
            archetypes[name] = {
                "candidate_id": row["candidate_id"],
                "final_score": round(row["final_score"], 4),
                "retrieval_score": round(row["retrieval_score"], 4),
                "consistency_score": round(row["consistency_score"], 4),
                "product_company_ratio": round(row["product_company_ratio"], 4),
                "honeypot_penalty": round(row["honeypot_penalty"], 4),
                "notice_period_days": int(row["notice_period_days"]) if "notice_period_days" in row else None,
                "response_rate": round(row["response_rate"], 4) if "response_rate" in row else None,
            }
        self.results["candidate_archetypes"] = archetypes

    # -------------------------------------------------------------------
    # Generate all plots
    # -------------------------------------------------------------------
    def generate_plots(self):
        pool = self.top1000
        # Final score distribution
        fig, ax = plt.subplots()
        ax.hist(pool["final_score"], bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel("Final Score")
        ax.set_title("Final Score Distribution (Top 1000)")
        save_plot(fig, "final_score_dist.png")

        # Retrieval score distribution
        if "retrieval_score" in pool.columns:
            fig, ax = plt.subplots()
            ax.hist(pool["retrieval_score"], bins=40, color="teal", edgecolor="white")
            ax.set_xlabel("Retrieval Score")
            ax.set_title("Retrieval Score Distribution")
            save_plot(fig, "retrieval_score_dist.png")

        # Consistency distribution
        if "consistency_score" in pool.columns:
            fig, ax = plt.subplots()
            ax.hist(pool["consistency_score"], bins=40, color="purple", edgecolor="white")
            ax.set_xlabel("Consistency Score")
            ax.set_title("Consistency Score Distribution")
            save_plot(fig, "consistency_dist.png")

        # Notice period distribution
        if "notice_period_days" in pool.columns:
            fig, ax = plt.subplots()
            ax.hist(pool["notice_period_days"], bins=30, color="orange", edgecolor="white")
            ax.set_xlabel("Notice Period (days)")
            ax.set_title("Notice Period Distribution")
            save_plot(fig, "notice_dist.png")

        # Experience distribution (merge with metadata)
        if "years_of_experience" in self.metadata.columns:
            merged = self.top1000[["candidate_id"]].merge(self.metadata, on="candidate_id", how="left")
            fig, ax = plt.subplots()
            ax.hist(merged["years_of_experience"].dropna(), bins=30, color="green", edgecolor="white")
            ax.set_xlabel("Years of Experience")
            ax.set_title("Experience Distribution (Top 1000)")
            save_plot(fig, "experience_dist.png")

        # Correlation heatmap
        feature_cols = [c for group in FEATURE_GROUPS.values() for c in group if c in pool.columns]
        if "final_score" in pool.columns:
            feature_cols.append("final_score")
        if len(feature_cols) > 1:
            corr = pool[feature_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
            ax.set_xticks(range(len(feature_cols)))
            ax.set_yticks(range(len(feature_cols)))
            ax.set_xticklabels(feature_cols, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(feature_cols, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Feature Correlation Heatmap", fontsize=14)
            save_plot(fig, "correlation_heatmap.png")

        # Additional: ranking curve and feature coverage
        self.ranking_curve_plot()
        self.feature_coverage_plot()

    # -------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------
    def save_reports(self):
        # Convert all numpy types in results to native Python
        safe_results = json.loads(json.dumps(self.results, default=json_safe))

        with open(REPORTS_DIR / "evaluation.json", "w") as f:
            json.dump(safe_results, f, indent=2)

        # Top 100 per-candidate CSV
        self.top1000.head(100).to_csv(REPORTS_DIR / "evaluation_top100.csv", index=False)

        # Pool statistics
        pool_stats = self.top1000.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()
        pool_stats.to_csv(REPORTS_DIR / "feature_statistics.csv")

        # Executive summary CSV
        ex = {
            "total_candidates": self.results["dataset"]["total_candidates"],
            "retrieved": self.results["dataset"]["retrieved_pool_size"],
            "final_shortlist": self.results["dataset"]["final_shortlist_size"],
            "avg_final_score": self.results["final_score"]["mean"],
            "avg_retrieval_score": self.results["retrieval"]["mean"],
            "avg_consistency": self.results["candidate_quality"]["consistency"]["mean"],
            "avg_notice_days": self.results["availability"]["notice_period_days"]["mean"],
            "open_to_work_pct": self.results["availability"]["open_to_work_pct"],
            "ranking_runtime_sec": self.results.get("ranking_runtime_sec", None),
            "validation": "PASS" if self.results["validation"]["all_passed"] else "FAIL",
        }
        pd.DataFrame([ex]).to_csv(REPORTS_DIR / "executive_summary.csv", index=False)

        # Pipeline summary JSON
        with open(REPORTS_DIR / "pipeline_summary.json", "w") as f:
            json.dump(ex, f, indent=2)

        # Markdown report
        md = self._generate_markdown()
        with open(REPORTS_DIR / "README_METRICS.md", "w") as f:
            f.write(md)

        print("All reports saved to reports/")

    def _generate_markdown(self) -> str:
        res = self.results
        md = "# Pipeline Evaluation Report\n\n"
        md += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        ds = res.get("dataset", {})
        md += "## Dataset\n"
        md += f"- Total candidates: **{ds.get('total_candidates', 'N/A'):,}**\n"
        md += f"- Retrieved: **{ds.get('retrieved_pool_size', 'N/A')}**\n"
        md += f"- Final shortlist: **{ds.get('final_shortlist_size', 'N/A')}**\n\n"

        ret = res.get("retrieval", {})
        md += "## Retrieval\n"
        md += f"- Mean: {ret.get('mean')}, Median: {ret.get('median')}, Std: {ret.get('std')}\n"
        md += f"- Range: [{ret.get('min')}, {ret.get('max')}]\n\n"

        fs = res.get("final_score", {})
        md += "## Final Score\n"
        md += f"- Mean: {fs.get('mean')}, Median: {fs.get('median')}, Std: {fs.get('std')}\n"
        md += f"- Range: [{fs.get('min')}, {fs.get('max')}]\n\n"

        rq = res.get("ranking_quality", {})
        md += "## Ranking Quality\n"
        md += f"- Top10 avg: {rq.get('top10_avg')}, Top100 avg: {rq.get('top100_avg')}\n"
        md += f"- Avg consecutive gap: {rq.get('avg_consecutive_gap')}\n\n"

        div = res.get("diversity", {})
        md += "## Diversity\n"
        md += f"- Unique companies: {div.get('unique_companies_in_top100')}\n"
        md += f"- Unique titles: {div.get('unique_titles_in_top100')}\n"
        md += f"- Most common: {div.get('most_represented_company')} ({div.get('most_represented_company_pct')}%)\n\n"

        arch = res.get("candidate_archetypes", {})
        md += "## Candidate Archetypes\n"
        for label, info in arch.items():
            md += f"- **{label.capitalize()}**: {info['candidate_id']} — Score {info['final_score']}, "
            md += f"Consistency {info['consistency_score']}, Notice {info.get('notice_period_days','N/A')}d\n"
        md += "\n"

        imp = res.get("feature_group_contribution_ablation", {})
        if imp:
            md += "## Feature Group Contribution (Ablation Analysis)\n"
            for grp, val in sorted(imp.items(), key=lambda x: -x[1]):
                md += f"- {grp}: {val:.1%}\n"
            md += "\n"

        ex = res.get("explainability", {})
        md += "## Explainability\n"
        md += f"- Coverage: {ex.get('coverage_pct')}%\n"
        md += f"- Avg length: {ex.get('avg_length_chars')} chars, {ex.get('avg_words')} words\n\n"

        val = res.get("validation", {})
        md += "## Validation\n"
        md += f"- Overall: {'✅ PASSED' if val.get('all_passed') else '❌ FAILED'}\n"
        for check, passed in val.get("checks", {}).items():
            md += f"- {check}: {'✓' if passed else '✗'}\n"
        md += "\n*All metrics from pipeline artifacts. No ground-truth labels required.*\n"
        return md


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluator = PipelineEvaluator()
    evaluator.load_data()
    evaluator.dataset_stats()
    evaluator.retrieval_stats()
    evaluator.final_score_stats()
    evaluator.ranking_quality()
    evaluator.diversity()
    evaluator.quality_stats()
    evaluator.availability_stats()
    evaluator.integrity_stats()
    evaluator.explainability_stats()
    evaluator.validation()
    evaluator.feature_group_contribution()
    evaluator.candidate_archetypes()
    evaluator.generate_plots()
    evaluator.save_reports()
    print("Evaluation complete.")
