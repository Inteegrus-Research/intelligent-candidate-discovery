#!/usr/bin/env python3
"""
Phase 6: consistency / honeypot scoring.

Produces:
- consistency_score: 0..1 (higher = more trustworthy)
- honeypot_penalty: 0..1 (higher = more suspicious)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import numpy as np

# --- domain lexicons ---
DOMAIN_KEYWORDS = {
    "software": {
        "software", "engineer", "backend", "frontend", "full stack", "full-stack",
        "api", "microservices", "spring", "django", "flask", "node", "react",
        "typescript", "javascript", "java", "go", "python", "cloud", "devops",
        "kubernetes", "docker", "distributed", "system design",
    },
    "data_ml": {
        "machine learning", "ml", "ai", "artificial intelligence", "llm", "rag",
        "embedding", "embeddings", "retrieval", "ranking", "recommendation",
        "nlp", "transformer", "fine tuning", "fine-tuning", "pytorch", "tensorflow",
        "xgboost", "lightgbm", "feature engineering", "data science",
    },
    "data_engineering": {
        "data pipeline", "data pipelines", "etl", "airflow", "spark", "hadoop",
        "dbt", "warehouse", "snowflake", "bigquery", "kafka", "flink", "beam",
        "databricks", "orchestration",
    },
    "marketing": {
        "marketing", "seo", "content", "brand", "campaign", "growth", "demand generation",
        "social", "copywriting", "editorial", "advertising",
    },
    "ops_pm": {
        "operations", "project manager", "project management", "product", "stakeholder",
        "process", "kpi", "delivery", "program management", "scrum", "agile", "ownership",
    },
    "finance": {
        "accounting", "finance", "financial", "tax", "audit", "gl", "ledger", "ind-as",
        "gaap", "cash flow", "reporting",
    },
    "design": {
        "design", "ui", "ux", "graphic", "figma", "illustrator", "photoshop", "branding",
    },
    "mechanical": {
        "mechanical", "cad", "solidworks", "creo", "ansys", "dfm", "dfma", "manufacturing",
        "hardware", "prototype",
    },
    "sales": {
        "sales", "account executive", "business development", "client", "revenue",
        "pipeline", "closing", "crm", "lead generation",
    },
    "hr": {
        "hr", "human resources", "talent", "recruiting", "people", "compensation",
        "payroll", "onboarding",
    },
    "support": {
        "support", "customer support", "customer success", "ticket", "escalation",
        "helpdesk", "sla",
    },
    "civil": {
        "civil", "construction", "site", "structural", "project site", "survey",
    },
}

SERVICE_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "ltimindtree", "l&t infotech",
    "deloitte", "pwc", "ey", "kpmg",
}

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+\-/]*", re.I)


def clean_text(x) -> str:
    if x is None:
        return ""
    x = str(x).lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(clean_text(text))


def text_has_any(text: str, phrases: Iterable[str]) -> bool:
    t = clean_text(text)
    return any(p in t for p in phrases)


def top_domains(text: str, top_k: int = 3) -> List[Tuple[str, float]]:
    t = clean_text(text)
    scores = []
    for domain, kws in DOMAIN_KEYWORDS.items():
        s = 0.0
        for kw in kws:
            if kw in t:
                # reward exact phrase hits more than single token hits
                s += 2.0 if " " in kw else 1.0
        if s > 0:
            scores.append((domain, s))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_k]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def weighted_domain_overlap(summary_text: str, career_text: str) -> float:
    """
    Returns 0..1 with higher meaning summary and career speak the same domain language.
    """
    summ = top_domains(summary_text, top_k=4)
    car = top_domains(career_text, top_k=4)

    summ_dom = [d for d, _ in summ]
    car_dom = [d for d, _ in car]

    if not summ_dom or not car_dom:
        return 0.5

    overlap = jaccard(summ_dom, car_dom)

    # If both sides are strong but disjoint, punish harder
    summ_strength = sum(s for _, s in summ)
    car_strength = sum(s for _, s in car)
    if overlap == 0.0 and summ_strength >= 2.0 and car_strength >= 2.0:
        return 0.0

    return float(np.clip(0.25 + 0.75 * overlap, 0.0, 1.0))


def compute_expert_zero_penalty(skills: List[dict]) -> float:
    """
    Penalize expert/advanced claims with zero evidence.
    """
    penalty = 0.0
    for s in skills or []:
        prof = clean_text(s.get("proficiency", ""))
        end = int(s.get("endorsements", 0) or 0)
        dur = int(s.get("duration_months", 0) or 0)
        if prof == "expert" and (end == 0 or dur == 0):
            penalty += 0.25
        elif prof == "advanced" and dur == 0:
            penalty += 0.10
    return float(np.clip(penalty, 0.0, 1.0))


def compute_yoe_career_penalty(yoe: float, career_years: float) -> float:
    """
    If career history exceeds stated YOE by a lot, suspicious.
    """
    if yoe is None or career_years is None:
        return 0.0
    gap = career_years - (yoe + 3.0)
    if gap <= 0:
        return 0.0
    # smooth penalty
    return float(np.clip(gap / 6.0, 0.0, 1.0))


def compute_low_yoe_many_advanced_penalty(yoe: float, advanced_skill_count: int, expert_skill_count: int) -> float:
    """
    Very low YOE + many advanced/expert claims = suspicious.
    """
    if yoe is None:
        return 0.0
    if yoe < 2.0 and (advanced_skill_count + expert_skill_count) > 5:
        return 0.35
    if yoe < 4.0 and (advanced_skill_count + expert_skill_count) > 8:
        return 0.20
    return 0.0


def compute_title_summary_career_penalty(current_title: str, summary_text: str, career_text: str) -> float:
    """
    Penalize when current title + summary drift strongly away from career evidence.
    """
    title = clean_text(current_title)
    summary = clean_text(summary_text)
    career = clean_text(career_text)

    summary_dom = top_domains(summary, top_k=3)
    career_dom = top_domains(career, top_k=3)
    title_dom = top_domains(title, top_k=2)

    summary_strength = sum(s for _, s in summary_dom)
    career_strength = sum(s for _, s in career_dom)
    title_strength = sum(s for _, s in title_dom)

    overlap = weighted_domain_overlap(summary, career)

    # If title says one thing, summary says another, and career says something else: bad.
    title_vs_career = jaccard([d for d, _ in title_dom], [d for d, _ in career_dom])

    penalty = 0.0
    if overlap == 0.0 and summary_strength >= 2.0 and career_strength >= 2.0:
        penalty += 0.35
    if title_vs_career == 0.0 and title_strength >= 1.0 and career_strength >= 2.0:
        penalty += 0.20

    # More direct: marketing-like summary vs mechanical/software career etc.
    if any(k in summary for k in ("marketing", "seo", "brand", "content", "copywriting")) and any(
        k in career for k in ("mechanical", "cad", "solidworks", "ansys", "software", "backend", "spark", "kafka")
    ):
        penalty += 0.25

    return float(np.clip(penalty, 0.0, 1.0))


def compute_honeypot_penalty(
    yoe: float,
    career_years: float,
    skills: List[dict],
    current_title: str,
    summary_text: str,
    career_text: str,
    advanced_skill_count: int,
    expert_skill_count: int,
) -> float:
    """
    Final suspicious-profile penalty, 0..1.
    """
    p1 = compute_yoe_career_penalty(yoe, career_years)
    p2 = compute_expert_zero_penalty(skills)
    p3 = compute_low_yoe_many_advanced_penalty(yoe, advanced_skill_count, expert_skill_count)
    p4 = compute_title_summary_career_penalty(current_title, summary_text, career_text)

    penalty = 0.35 * p1 + 0.25 * p2 + 0.20 * p3 + 0.20 * p4
    return float(np.clip(penalty, 0.0, 1.0))


def compute_consistency_score(
    yoe: float,
    career_years: float,
    skills: List[dict],
    current_title: str,
    summary_text: str,
    career_text: str,
    advanced_skill_count: int,
    expert_skill_count: int,
) -> dict:
    """
    Returns both consistency_score and honeypot_penalty.
    """
    yoe = float(yoe or 0.0)
    career_years = float(career_years or 0.0)
    summary_text = clean_text(summary_text)
    career_text = clean_text(career_text)
    current_title = clean_text(current_title)

    honeypot_penalty = compute_honeypot_penalty(
        yoe=yoe,
        career_years=career_years,
        skills=skills,
        current_title=current_title,
        summary_text=summary_text,
        career_text=career_text,
        advanced_skill_count=advanced_skill_count,
        expert_skill_count=expert_skill_count,
    )

    # components in 0..1
    yoe_gap = abs(yoe - career_years)
    yoe_alignment = float(np.clip(1.0 - min(yoe_gap / max(max(yoe, career_years, 1.0), 1.0), 1.0), 0.0, 1.0))
    domain_alignment = weighted_domain_overlap(summary_text, career_text)

    # skill evidence strength
    skill_evidence = 0.0
    strong_count = 0
    for s in skills or []:
        prof = clean_text(s.get("proficiency", ""))
        end = int(s.get("endorsements", 0) or 0)
        dur = int(s.get("duration_months", 0) or 0)
        if prof in {"advanced", "expert"}:
            strong_count += 1
        skill_evidence += min(1.0, (0.25 if prof == "beginner" else 0.5 if prof == "intermediate" else 0.8 if prof == "advanced" else 1.0))
        skill_evidence += min(0.15, math.log1p(end) / 20.0)
        skill_evidence += min(0.15, math.log1p(dur) / 25.0)

    skill_evidence = float(np.clip(skill_evidence / max(len(skills) or 1, 1), 0.0, 1.0))
    strong_skill_penalty = 0.0 if strong_count == 0 else 0.0

    consistency_score = (
        0.35 * yoe_alignment +
        0.30 * domain_alignment +
        0.20 * skill_evidence +
        0.15 * (1.0 - honeypot_penalty) -
        strong_skill_penalty
    )

    consistency_score = float(np.clip(consistency_score, 0.0, 1.0))

    return {
        "consistency_score": consistency_score,
        "honeypot_penalty": honeypot_penalty,
        "yoe_alignment_score": yoe_alignment,
        "domain_alignment_score": domain_alignment,
        "skill_evidence_score": skill_evidence,
    }
