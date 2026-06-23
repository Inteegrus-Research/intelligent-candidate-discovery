import numpy as np
from features.common import JD_CORE_SKILLS, normalize_skill, proficiency_score, relevant_core_skill

def extract_skill_features(skills, skill_assessment_scores):
    skill_overlap = 0
    weighted_skill_overlap = 0.0
    advanced_skill_count = 0
    expert_skill_count = 0
    matched_assessments = []

    seen_core = set()
    for s in skills or []:
        name = normalize_skill(s.get("name", ""))
        prof = str(s.get("proficiency", "")).lower()
        end = int(s.get("endorsements", 0) or 0)
        dur = int(s.get("duration_months", 0) or 0)

        if prof in {"advanced", "expert"}:
            advanced_skill_count += 1
        if prof == "expert":
            expert_skill_count += 1

        core = relevant_core_skill(name)
        if core and core not in seen_core:
            seen_core.add(core)
            skill_overlap += 1
            weighted_skill_overlap += proficiency_score(prof) * np.log1p(end) * np.log1p(dur)

    if isinstance(skill_assessment_scores, dict) and skill_assessment_scores:
        for k, v in skill_assessment_scores.items():
            core = relevant_core_skill(k)
            if core:
                matched_assessments.append(float(v))
        if not matched_assessments:
            matched_assessments = [float(v) for v in skill_assessment_scores.values()]
    assessment_score = float(np.mean(matched_assessments)) if matched_assessments else 0.0

    return {
        "skill_overlap": skill_overlap,
        "weighted_skill_overlap": weighted_skill_overlap,
        "advanced_skill_count": advanced_skill_count,
        "expert_skill_count": expert_skill_count,
        "assessment_score": assessment_score,
        "skill_coverage_ratio": skill_overlap / max(len(JD_CORE_SKILLS), 1),
    }
