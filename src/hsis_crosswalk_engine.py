"""Single entry point: ingest -> LLM normalize -> SBERT embed -> same-File pair scoring."""

import argparse
from pathlib import Path

import pandas as pd

from llm_normalizer import normalize_metadata
from pair_scorer import (
    ACCEPT_THRESHOLD,
    REVIEW_THRESHOLD,
    build_embedding_text,
    score_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw_states" / "hsis_raw_schema.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "step3_candidate_matches.csv"
REQUIRED_COLUMNS = ["State", "File", "Variable_Name", "Description", "Field_Type"]


def load_input(path: Path, exclude_states: list[str]) -> pd.DataFrame:
    """Read and validate the raw data dictionary."""
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].apply(lambda col: col.str.strip())
    df = df[df["Variable_Name"] != ""]
    df = df.drop_duplicates(subset=["State", "File", "Variable_Name"])

    if exclude_states:
        excluded = {s.upper() for s in exclude_states}
        df = df[~df["State"].str.upper().isin(excluded)]

    return df.reset_index(drop=True)


def normalize_all(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    """Run LLM (or offline) normalization on every row and build embedding text."""
    use_llm = provider != "offline"
    texts, sources = [], []
    total = len(df)

    for n, row in enumerate(df.itertuples(index=False), start=1):
        normalized, source, cache_hit = normalize_metadata(
            row.State,
            row.Variable_Name,
            row.Description,
            row.Field_Type,
            use_llm=use_llm,
            llm_provider=provider if use_llm else "gemini",
        )
        texts.append(build_embedding_text(row.File, normalized, row.Field_Type))
        sources.append(source)
        print(
            f"[{n}/{total}] {row.State}/{row.File}/{row.Variable_Name} "
            f"({'cache' if cache_hit else source})"
        )

    df = df.copy()
    df["embedding_text"] = texts
    df["normalization_source"] = sources
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="HSIS cross-state metadata matcher")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--llm_provider",
        choices=["gemini", "openai", "offline"],
        default="gemini",
        help="Normalization backend. 'offline' makes no API calls.",
    )
    parser.add_argument("--accept", type=float, default=ACCEPT_THRESHOLD)
    parser.add_argument("--review", type=float, default=REVIEW_THRESHOLD)
    parser.add_argument(
        "--exclude_states",
        nargs="*",
        default=[],
        help="State codes to skip, e.g. --exclude_states USAH USAN",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Drop pairs below this score from the output file (shrinks it).",
    )
    args = parser.parse_args()

    df = load_input(args.input, args.exclude_states)
    print(f"Loaded {len(df)} variables from {df['State'].nunique()} states.")

    df = normalize_all(df, args.llm_provider)
    results = score_pairs(df, accept=args.accept, review=args.review)
    results = results[results["cosine_similarity"] >= args.min_score]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(f"\nWrote {len(results)} pairs to {args.output}")
    print(results["decision"].value_counts().to_string())


if __name__ == "__main__":
    main()