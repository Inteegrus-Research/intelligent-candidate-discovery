import numpy as np
from scoring.consistency_engine import compute_honeypot_penalty, compute_profile_consistency_score

def extract_quality_features(base_row, sig, skills, career_history):
    yoe = float(base_row.get("yoe", 0) or 0)
    career_years = float(base_row.get("career_years_from_history", 0) or 0)
    title_match = float(base_row.get("title_career_match_ratio", 0) or 0)

    github_raw = float(sig.get("github_activity_score", -1) or -1)
    github_missing = int(github_raw == -1)
    github_activity = 0.0 if github_missing else github_raw

    profile_completeness = float(sig.get("profile_completeness_score", 0) or 0)
    connections = int(sig.get("connection_count", 0) or 0)
    endorsements = int(sig.get("endorsements_received", 0) or 0)

    penalty = compute_honeypot_penalty(
        yoe=yoe,
        career_years=career_years,
        skills=skills,
        skill_assessment_scores=sig.get("skill_assessment_scores", {}),
        title_career_match_ratio=title_match,
    )
    consistency = compute_profile_consistency_score(yoe, career_years, title_match, penalty)

    quality_score = (
        0.30 * (profile_completeness / 100.0) +
        0.20 * np.log1p(connections) / np.log1p(1000) +
        0.20 * np.log1p(endorsements) / np.log1p(1000) +
        0.15 * (github_activity / 100.0) +
        0.15 * consistency
    )

    return {
        "github_activity": github_activity,
        "github_activity_missing": github_missing,
        "profile_completeness": profile_completeness,
        "connection_count": connections,
        "endorsements_received": endorsements,
        "profile_consistency_score": consistency,
        "honeypot_penalty": penalty,
        "quality_score": float(quality_score),
    }
