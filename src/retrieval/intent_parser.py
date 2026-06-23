#!/usr/bin/env python3
"""
Parse the JD into 5 intent chunks for multi-intent retrieval.
"""

import re
from typing import Dict, List


INTENT_KEYWORDS = {
    "technical_search_ranking": [
        "retrieval", "ranking", "recommendation", "search", "vector search",
        "learning-to-rank", "bm25", "hybrid search", "rerank"
    ],
    "ml_engineering": [
        "production ml", "pipelines", "deployment", "monitoring",
        "inference", "evaluation", "scalable", "latency", "production"
    ],
    "llm_infrastructure": [
        "embeddings", "vector db", "vector database", "fine tuning",
        "fine-tuning", "rag", "llm", "sentence-transformers", "openai",
        "bge", "e5", "faiss", "milvus", "qdrant", "weaviate"
    ],
    "product_mindset": [
        "product company", "ownership", "shipping", "impact",
        "recruiter workflows", "eval frameworks", "ship a working ranker",
        "working ranker"
    ],
    "availability": [
        "location", "notice period", "active", "open to work",
        "pune", "noida", "hybrid", "relocate", "recruiter response rate",
        "last active"
    ],
}


def clean_text(x: str) -> str:
    if not x:
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def sentence_split(text: str) -> List[str]:
    text = clean_text(text)
    # Simple and robust enough for the JD document
    sents = re.split(r"(?<=[.!?])\s+(?=[a-z0-9(])", text)
    return [s.strip() for s in sents if s.strip()]


def pick_sentences(sentences: List[str], keywords: List[str], max_sentences: int = 6) -> List[str]:
    picked = []
    for s in sentences:
        if any(k in s for k in keywords):
            picked.append(s)
        if len(picked) >= max_sentences:
            break
    # de-duplicate while keeping order
    deduped = []
    seen = set()
    for s in picked:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped[:max_sentences]


def build_jd_intents(jd_text: str) -> Dict[str, str]:
    """
    Return 5 intent strings keyed by intent name.
    """
    sentences = sentence_split(jd_text)

    intents = {}

    # 1) Technical Search / Ranking
    s1 = pick_sentences(sentences, INTENT_KEYWORDS["technical_search_ranking"])
    intents["technical_search_ranking"] = (
        "technical search and ranking intent. "
        + " ".join(s1)
    ).strip()

    # 2) ML Engineering
    s2 = pick_sentences(sentences, INTENT_KEYWORDS["ml_engineering"])
    intents["ml_engineering"] = (
        "production ml engineering intent. "
        + " ".join(s2)
    ).strip()

    # 3) LLM Infrastructure
    s3 = pick_sentences(sentences, INTENT_KEYWORDS["llm_infrastructure"])
    intents["llm_infrastructure"] = (
        "llm infrastructure intent. "
        + " ".join(s3)
    ).strip()

    # 4) Product Mindset
    s4 = pick_sentences(sentences, INTENT_KEYWORDS["product_mindset"])
    intents["product_mindset"] = (
        "product mindset and execution intent. "
        + " ".join(s4)
    ).strip()

    # 5) Availability
    s5 = pick_sentences(sentences, INTENT_KEYWORDS["availability"])
    intents["availability"] = (
        "availability and hiring fit intent. "
        + " ".join(s5)
    ).strip()

    # Fallbacks if a bucket is too empty
    for key, text in intents.items():
        if len(text.split()) < 12:
            intents[key] = f"{key.replace('_', ' ')}. {jd_text[:1200]}"

    return intents
