"""Embed normalized metadata, build same-File state pairs, score and route them."""

from itertools import combinations

import pandas as pd
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
ACCEPT_THRESHOLD = 0.82
REVIEW_THRESHOLD = 0.70


def build_embedding_text(file_name: str, normalized, field_type: str) -> str:
    """One combined text per variable. Field_Type is context, not a filter."""
    return (
        f"File: {file_name}. "
        f"Name: {normalized.standard_name}. "
        f"Domain: {normalized.domain_category}. "
        f"Description: {normalized.description}. "
        f"Unit: {normalized.unit_type}. "
        f"Field type: {field_type}."
    )


def route(score: float, accept: float, review: float) -> str:
    if score >= accept:
        return "accepted"
    if score >= review:
        return "hitl_review"
    return "rejected"


def candidate_blocks(df: pd.DataFrame):
    """Yield (file, state_a, state_b, idx_a, idx_b) for each state pair
    within the same File. Pairs never cross Files."""
    for file_name, file_df in df.groupby("File"):
        states = sorted(file_df["State"].unique())
        for state_a, state_b in combinations(states, 2):
            idx_a = file_df.index[file_df["State"] == state_a].tolist()
            idx_b = file_df.index[file_df["State"] == state_b].tolist()
            yield file_name, state_a, state_b, idx_a, idx_b


def score_pairs(
    df: pd.DataFrame,
    accept: float = ACCEPT_THRESHOLD,
    review: float = REVIEW_THRESHOLD,
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """df needs a default RangeIndex and an 'embedding_text' column."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["embedding_text"].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    records = []
    for file_name, state_a, state_b, idx_a, idx_b in candidate_blocks(df):
        sims = util.cos_sim(embeddings[idx_a], embeddings[idx_b]).tolist()
        for i, row_a in enumerate(idx_a):
            for j, row_b in enumerate(idx_b):
                score = sims[i][j]
                a, b = df.loc[row_a], df.loc[row_b]
                records.append({
                    "file": file_name,
                    "state_a": state_a,
                    "variable_a": a["Variable_Name"],
                    "description_a": a["Description"],
                    "state_b": state_b,
                    "variable_b": b["Variable_Name"],
                    "description_b": b["Description"],
                    "text_a": a["embedding_text"],
                    "text_b": b["embedding_text"],
                    "cosine_similarity": round(score, 4),
                    "decision": route(score, accept, review),
                    "reviewer_status": "pending" if review <= score < accept else "",
                    "reviewer_notes": "",
                })

    result = pd.DataFrame(records)
    return result.sort_values("cosine_similarity", ascending=False).reset_index(drop=True)