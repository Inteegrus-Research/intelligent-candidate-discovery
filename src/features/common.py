import gzip
import json
import math
import re
from datetime import date, datetime
from pathlib import Path

SERVICE_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "mindtree", "hcl", "tech mahindra", "ltimindtree", "l&t infotech",
    "deloitte", "pwc", "ey", "kpmg"
}

TIER1_CITIES = {
    "pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
    "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai"
}

JD_CORE_SKILLS = {
    "retrieval", "ranking", "recommendation systems", "search", "vector db",
    "embeddings", "llm", "fine-tuning llms", "learning-to-rank",
    "sentence-transformers", "faiss", "milvus", "weaviate", "qdrant",
    "elasticsearch", "opensearch", "python", "pytorch", "tensorflow",
    "spark", "airflow", "nlp", "production ml", "pipelines",
    "deployment", "monitoring", "xgboost", "rerank", "bm25", "hybrid search"
}

PROF_MAP = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.80,
    "expert": 1.00,
}

SKILL_SYNONYMS = {
    "sentence transformers": "sentence-transformers",
    "sentence-transformer": "sentence-transformers",
    "vector database": "vector db",
    "vector databases": "vector db",
    "vector store": "vector db",
    "vector stores": "vector db",
    "llms": "llm",
    "large language model": "llm",
    "large language models": "llm",
    "fine tuning llms": "fine-tuning llms",
    "fine-tuning llms": "fine-tuning llms",
    "learning to rank": "learning-to-rank",
    "recommendation system": "recommendation systems",
    "recommendation systems": "recommendation systems",
}

SEARCH_RE = re.compile(
    r"\b(ranking|retrieval|recommendation|search|vector search|relevance|personalization)\b",
    re.I,
)
BUILD_RE = re.compile(r"\b(built|shipped|deployed|owned|launched|designed|implemented)\b", re.I)


def clean_text(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_skill(name):
    x = clean_text(name).replace("&", " and ")
    x = re.sub(r"[^a-z0-9+\-/ ]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return SKILL_SYNONYMS.get(x, x)


def proficiency_score(p):
    return PROF_MAP.get(clean_text(p), 0.0)


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception:
        return None


def days_ago(d, today):
    return (today - d).days if d else None


def load_raw_candidates(path):
    path = Path(path)
    if str(path).endswith(".gz"):
        f = gzip.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r", encoding="utf-8")
    with f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_tier1_or_flexible_location(location, willing_to_relocate):
    loc = clean_text(location)
    if willing_to_relocate:
        return 1
    return int(any(city in loc for city in TIER1_CITIES))


def relevant_core_skill(name):
    n = normalize_skill(name)
    for core in JD_CORE_SKILLS:
        if n == core or core in n or n in core:
            return core
    return None
