"""Build a draft master concept list from cached Gemini normalizations."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_normalizer import build_cache_key, load_cache

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT = PROJECT_ROOT / "data" / "raw_states" / "hsis_raw_schema.csv"
OUT_VARS = PROJECT_ROOT / "outputs" / "variable_normalizations.csv"
OUT_NORMALIZATIONS_JSON = PROJECT_ROOT / "data" / "master_schema" / "variable_normalizations.json"
OUT_MASTER = PROJECT_ROOT / "data" / "master_schema" / "hsis_master_concepts_draft.csv"


def read_input() -> pd.DataFrame:
    try:
        df = pd.read_csv(INPUT, dtype=str, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT, dtype=str, encoding="cp1252")
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    cols = ["State", "File", "Variable_Name", "Description", "Field_Type"]
    df = df[cols].apply(lambda c: c.str.strip())
    df = df[df["Variable_Name"] != ""]
    return df.drop_duplicates(subset=["State", "File", "Variable_Name"]).reset_index(drop=True)


def most_common(series: pd.Series) -> str:
    return series.mode().iloc[0]


def main() -> None:
    df = read_input()
    cache = load_cache()

    records, missing = [], 0
    for row in df.to_dict("records"):
        key = build_cache_key(
            row["State"], row["Variable_Name"], row["Description"],
            row["Field_Type"], "gemini", "auto",
        )
        if key not in cache:
            missing += 1
            continue
        norm = cache[key]["normalized"]
        records.append({**row, **norm})

    print(f"Input variables: {len(df)} | normalized: {len(records)} | missing from cache: {missing}")
    vars_df = pd.DataFrame(records)
    OUT_VARS.parent.mkdir(parents=True, exist_ok=True)
    vars_df.to_csv(OUT_VARS, index=False)
    vars_df.to_json(
        OUT_NORMALIZATIONS_JSON,
        orient="records",
        indent=2,
        force_ascii=False,
    )
    print(f"Wrote readable normalizations to {OUT_NORMALIZATIONS_JSON}")
    concepts = []
    for (file_name, name), g in vars_df.groupby(["File", "standard_name"]):
        concepts.append({
            "master_concept_id": f"{file_name.lower()}.{name}",
            "file": file_name,
            "standard_name": name,
            "domain_category": most_common(g["domain_category"]),
            "description": most_common(g["description"]),
            "unit_type": most_common(g["unit_type"]),
            "n_states": g["State"].nunique(),
            "source_variables": "; ".join(
                f"{s}:{v}" for s, v in zip(g["State"], g["Variable_Name"])
            ),
        })

    master = pd.DataFrame(concepts).sort_values("n_states", ascending=False)
    OUT_MASTER.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_MASTER, index=False)
    print(f"Wrote {len(master)} draft concepts to {OUT_MASTER}")
    print(f"Concepts found in 2+ states: {(master['n_states'] >= 2).sum()}")


if __name__ == "__main__":
    main()