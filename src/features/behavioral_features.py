import numpy as np
from features.common import parse_date, days_ago

def extract_behavioral_features(sig, as_of_date):
    last_active = parse_date(sig.get("last_active_date"))
    days_active = days_ago(last_active, as_of_date) if last_active else None
    response_rate = float(sig.get("recruiter_response_rate", 0) or 0)
    interview_completion = float(sig.get("interview_completion_rate", 0) or 0)

    views = int(sig.get("profile_views_received_30d", 0) or 0)
    saved = int(sig.get("saved_by_recruiters_30d", 0) or 0)
    search = int(sig.get("search_appearance_30d", 0) or 0)
    avg_resp = float(sig.get("avg_response_time_hours", 0) or 0)

    return {
        "days_active": days_active if days_active is not None else 9999,
        "activity_recency_score": float(np.exp(-(days_active or 9999) / 90.0)),
        "response_rate": response_rate,
        "interview_completion": interview_completion,
        "profile_views": float(np.log1p(views)),
        "saved_by_recruiters": float(np.log1p(saved)),
        "search_appearance": float(np.log1p(search)),
        "open_to_work": int(bool(sig.get("open_to_work_flag"))),
        "response_speed_score": float(np.exp(-(avg_resp or 9999) / 72.0)),
    }
