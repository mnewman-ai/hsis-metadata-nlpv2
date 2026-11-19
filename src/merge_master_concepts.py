"""Cluster variables into master concepts by embedding similarity (same File only)."""

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_VARS = PROJECT_ROOT / "outputs" / "variable_normalizations.csv"
OUT_MASTER = PROJECT_ROOT / "data" / "master_schema" / "hsis_master_concepts.csv"
OUT_MAP = PROJECT_ROOT / "outputs" / "variable_concepts.csv"
MERGE_THRESHOLD = 0.82  # cosine similarity needed to merge


def main() -> None:
    df = pd.read_csv(IN_VARS, dtype=str).fillna("")
    df["text"] = df["standard_name"].str.replace("_", " ") + ": " + df["description"]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    df["concept_id"] = ""

    for file_name, g in df.groupby("File"):
        emb = model.encode(g["text"].tolist(), normalize_embeddings=True)
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1 - MERGE_THRESHOLD,
        ).fit_predict(emb)
        df.loc[g.index, "concept_id"] = [f"{file_name.lower()}_{l:04d}" for l in labels]

    concepts = []
    for cid, g in df.groupby("concept_id"):
        concepts.append({
            "master_concept_id": cid,
            "file": g["File"].iloc[0],
            "standard_name": g["standard_name"].mode().iloc[0],
            "domain_category": g["domain_category"].mode().iloc[0],
            "description": g["description"].mode().iloc[0],
            "unit_type": g["unit_type"].mode().iloc[0],
            "n_states": g["State"].nunique(),
            "n_variables": len(g),
            "source_variables": "; ".join(f"{s}:{v}" for s, v in zip(g["State"], g["Variable_Name"])),
        })

    master = pd.DataFrame(concepts).sort_values("n_states", ascending=False)
    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_MASTER, index=False)
    df.drop(columns=["text"]).to_csv(OUT_MAP, index=False)

    print(f"{len(df)} variables -> {len(master)} master concepts")
    print(f"Concepts in 2+ states: {(master.n_states >= 2).sum()}")
    print(f"Single-state concepts: {(master.n_states == 1).sum()}")


if __name__ == "__main__":
    main()