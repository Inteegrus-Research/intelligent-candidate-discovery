import numpy as np
import pandas as pd

def compute_semantic_features(candidate_embeddings_path, jd_intent_embeddings_path, candidate_ids, batch_size=5000):
    cand = np.load(candidate_embeddings_path, mmap_mode="r")
    jd = np.load(jd_intent_embeddings_path, mmap_mode="r").astype(np.float32)

    n = cand.shape[0]
    out = {
        "candidate_id": candidate_ids,
        "profile_similarity": np.zeros(n, dtype=np.float32),
        "skills_similarity": np.zeros(n, dtype=np.float32),
        "career_similarity": np.zeros(n, dtype=np.float32),
        "full_similarity": np.zeros(n, dtype=np.float32),
        "intent1_similarity": np.zeros(n, dtype=np.float32),
        "intent2_similarity": np.zeros(n, dtype=np.float32),
        "intent3_similarity": np.zeros(n, dtype=np.float32),
        "intent4_similarity": np.zeros(n, dtype=np.float32),
        "intent5_similarity": np.zeros(n, dtype=np.float32),
        "semantic_alignment_score": np.zeros(n, dtype=np.float32),
        "intent_weighted_full_similarity": np.zeros(n, dtype=np.float32),
    }

    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = np.asarray(cand[start:end], dtype=np.float32)  # (B, 4, 384)
        sims = np.einsum("bvd,id->bvi", chunk, jd)             # (B, 4, 5)

        mean_doc = sims.mean(axis=2)                           # (B, 4)
        out["profile_similarity"][start:end] = mean_doc[:, 0]
        out["skills_similarity"][start:end] = mean_doc[:, 1]
        out["career_similarity"][start:end] = mean_doc[:, 2]
        out["full_similarity"][start:end] = mean_doc[:, 3]

        out["intent1_similarity"][start:end] = sims[:, 3, 0]
        out["intent2_similarity"][start:end] = sims[:, 3, 1]
        out["intent3_similarity"][start:end] = sims[:, 3, 2]
        out["intent4_similarity"][start:end] = sims[:, 3, 3]
        out["intent5_similarity"][start:end] = sims[:, 3, 4]

        out["semantic_alignment_score"][start:end] = (
            0.25 * mean_doc[:, 0] + 0.25 * mean_doc[:, 1] + 0.25 * mean_doc[:, 2] + 0.25 * mean_doc[:, 3]
        )
        out["intent_weighted_full_similarity"][start:end] = (sims[:, 3, :] * weights).sum(axis=1)

    return pd.DataFrame(out)
