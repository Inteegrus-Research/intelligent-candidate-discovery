from features.common import clean_text, is_tier1_or_flexible_location

def extract_availability_features(sig, profile):
    notice_days = int(sig.get("notice_period_days", 90) or 90)
    notice_score = 1.0 - min(notice_days, 90) / 90.0

    offer_raw = sig.get("offer_acceptance_rate", -1)
    offer_missing = int(offer_raw is None or float(offer_raw) == -1)
    offer_acceptance = 0.5 if offer_missing else float(offer_raw)

    location = profile.get("location", "")
    willing = int(bool(sig.get("willing_to_relocate")))
    location_match = is_tier1_or_flexible_location(location, willing)

    preferred_work_mode = clean_text(sig.get("preferred_work_mode", ""))
    work_mode_match = int(preferred_work_mode in {"hybrid", "onsite", "flexible"})

    return {
        "notice_score": notice_score,
        "notice_period_days_clean": notice_days,
        "verified_email": int(bool(sig.get("verified_email"))),
        "verified_phone": int(bool(sig.get("verified_phone"))),
        "linkedin_connected": int(bool(sig.get("linkedin_connected"))),
        "offer_acceptance": offer_acceptance,
        "offer_acceptance_missing": offer_missing,
        "location_match": location_match,
        "work_mode_match": work_mode_match,
        "willing_to_relocate": willing,
    }
