import re
from features.common import SEARCH_RE, BUILD_RE, clean_text

def extract_career_features(career_history, profile):
    parts = []
    for r in career_history or []:
        parts.append(clean_text(r.get("title", "")))
        parts.append(clean_text(r.get("description", "")))
        parts.append(clean_text(r.get("company", "")))
    text = " ".join(parts)

    search_count = len(SEARCH_RE.findall(text))
    build_count = len(BUILD_RE.findall(text))

    return {
        "has_search_experience": int(search_count > 0),
        "search_experience_count": search_count,
        "career_evidence_score": min(search_count / 3.0, 1.0),
        "build_ownership_count": build_count,
    }
