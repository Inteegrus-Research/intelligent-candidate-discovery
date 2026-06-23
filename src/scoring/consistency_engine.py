import numpy as np
from features.common import normalize_skill

def compute_honeypot_penalty(yoe, career_years, skills, skill_assessment_scores, title_career_match_ratio):
    penalty = 0.0

    if yoe is not None and career_years is not None and career_years > yoe + 3:
        penalty += 0.35

    expert_zero = False
    advanced_count = 0
    for s in skills or []:
        prof = str(s.get("proficiency", "")).lower()
        end = int(s.get("endorsements", 0) or 0)
        dur = int(s.get("duration_months", 0) or 0)
        if prof in {"advanced", "expert"}:
            advanced_count += 1
        if prof == "expert" and (end == 0 or dur == 0):
            expert_zero = True

    if expert_zero:
        penalty += 0.25
    if (yoe or 0) < 2 and advanced_count > 5:
        penalty += 0.25
    if title_career_match_ratio is not None and title_career_match_ratio < 0.15 and (yoe or 0) > 5:
        penalty += 0.15

    return min(1.0, penalty)


def compute_profile_consistency_score(yoe, career_years, title_career_match_ratio, honeypot_penalty):
    yoe = float(yoe or 0)
    career_years = float(career_years or 0)
    gap = abs(yoe - career_years) / max(max(yoe, career_years, 1.0), 1.0)
    yoe_score = 1.0 - min(gap, 1.0)
    title_score = float(title_career_match_ratio or 0)
    return float(np.clip(0.45 * yoe_score + 0.35 * title_score + 0.20 * (1.0 - honeypot_penalty), 0.0, 1.0))
