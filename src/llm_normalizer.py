"""Normalize state DOT metadata into a structured representation.

Supports Gemini (with automatic model fallback) and OpenAI, and caches
normalized results locally. Use `use_llm=False` for an offline, no-API fallback.
"""

import hashlib
import json
import os
import re
from pathlib import Path
import time

from openai import OpenAI
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / "data" / "cache" / "llm_normalizations.json"
PROMPT_VERSION = "1.0"

DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-4o-mini",
}

VALID_PROVIDERS = ("gemini", "openai")

# Tried in order, newest first. Add or remove names as models change.
# Try text-generation models from newest/preferred to older/lighter options.
# Image-only, embedding, research, and experimental non-chat models are excluded.
GEMINI_MODEL_CHAIN = [
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

_last_gemini_model = None
GEMINI_BATCH_SIZE = 10
ACRONYM_LOOKUP = {
    "AADT": "annual average daily traffic",
    "ADT": "average daily traffic",
    "LNWID": "lane width",
    "LANE_WD": "lane width",
    "LANE_WIDTH_NBR": "lane width number",
    "SPD": "speed",
    "SPEEDLIMIT": "posted speed limit",
    "DESG_SPD": "design speed",
    "SHLD": "shoulder",
    "MED": "median",
}

SYSTEM_PROMPT = """
You are an expert Transportation Data Architecture Engine.

Normalize one raw state DOT data-dictionary field for semantic matching
against fields from other states' highway-safety data dictionaries.

Return only JSON with exactly these keys:
- standard_name
- domain_category
- description
- unit_type

Rules:
- Expand abbreviations when their meaning is reasonably supported.
- Do not invent units, meanings, transformations, or source facts.
- Use "not specified" when the unit cannot be determined.
- Keep descriptions concise and factual.
""".strip()


class NormalizedMetadata(BaseModel):
    """The JSON schema required by the project prompt design."""

    standard_name: str
    domain_category: str
    description: str
    unit_type: str

class BatchNormalizedMetadata(BaseModel):
    """Structured response for a batch of metadata records."""

    items: list[NormalizedMetadata]

def clean_text(value: object) -> str:
    """Convert empty/missing values to a clean string."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def expand_acronyms(text: str) -> str:
    """Expand known HSIS/state-DOT acronyms without changing unknown tokens."""
    expanded = clean_text(text)

    for acronym, meaning in ACRONYM_LOOKUP.items():
        expanded = re.sub(
            rf"\b{re.escape(acronym)}\b",
            meaning,
            expanded,
            flags=re.IGNORECASE,
        )

    return expanded


def load_cache() -> dict:
    """Load prior normalizations; return an empty cache if none exists."""
    if not CACHE_FILE.exists():
        return {}

    try:
        with CACHE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict) -> None:
    """Write the normalization cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CACHE_FILE.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, sort_keys=True)


def build_cache_key(
    state: str,
    variable_name: str,
    description: str,
    field_type: str,
    provider: str,
    model: str,
) -> str:
    """Create a stable key so only changed fields cause a new API call."""
    source = {
        "prompt_version": PROMPT_VERSION,
        "state": clean_text(state),
        "variable_name": clean_text(variable_name),
        "description": clean_text(description),
        "field_type": clean_text(field_type),
        "normalization_mode": provider,
        "model": model,
    }

    serialized = json.dumps(source, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def offline_normalization(
    variable_name: str,
    description: str,
    field_type: str,
) -> NormalizedMetadata:
    """Create a deterministic fallback without an API request."""
    expanded_name = expand_acronyms(variable_name)
    expanded_description = expand_acronyms(description)

    return NormalizedMetadata(
        standard_name=expanded_name.lower().replace(" ", "_"),
        domain_category="highway_safety_and_roadway_metadata",
        description=expanded_description or expanded_name,
        unit_type=clean_text(field_type) or "not specified",
    )


def build_user_prompt(state, variable_name, description, field_type) -> str:
    return f"""
State: {clean_text(state)}
Raw variable name: {clean_text(variable_name)}
Known abbreviation expansion: {expand_acronyms(variable_name)}
Source description: {clean_text(description)}
Field type: {clean_text(field_type)}
""".strip()


def get_model_name(provider: str) -> str:
    """Model comes from <PROVIDER>_MODEL if set, else the default."""
    return os.environ.get(f"{provider.upper()}_MODEL", DEFAULT_MODELS[provider])

GEMINI_MIN_SECONDS_BETWEEN_CALLS = 4.5
_last_gemini_call = 0.0


def _throttle() -> None:
    """Wait so real Gemini requests start at least N seconds apart."""
    global _last_gemini_call
    wait = GEMINI_MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_gemini_call)
    if wait > 0:
        time.sleep(wait)
    _last_gemini_call = time.monotonic()

def call_gemini(user_prompt: str, model_name: str) -> NormalizedMetadata:
    _throttle()
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=NormalizedMetadata,
        ),
    )

    if response.parsed is not None:
        return response.parsed
    return NormalizedMetadata.model_validate_json(response.text)


def call_gemini_with_fallback(user_prompt: str):
    """Try each Gemini model until one works. Returns (result, model_used)."""
    global _last_gemini_model

    override = os.environ.get("GEMINI_MODEL")
    chain = [override] if override else GEMINI_MODEL_CHAIN

    last_error = None
    for model_name in chain:
        try:
            result = call_gemini(user_prompt, model_name)
        except Exception as exc:
            if not any(
                code in str(exc)
                for code in ("503", "404", "429", "UNAVAILABLE", "NOT_FOUND")
            ):
                raise
            print(f"  {model_name} unavailable, trying next model...")
            last_error = exc
            continue

        if model_name != _last_gemini_model:
            print(f"  Using Gemini model: {model_name}")
            _last_gemini_model = model_name
        return result, model_name

    raise RuntimeError("All Gemini models in the chain failed.") from last_error

def build_batch_user_prompt(records: list[dict]) -> str:
    """Build one prompt containing several metadata records."""
    items = []

    for index, record in enumerate(records):
        items.append(
            {
                "index": index,
                "state": clean_text(record["State"]),
                "variable_name": clean_text(record["Variable_Name"]),
                "known_abbreviation_expansion": expand_acronyms(
                    record["Variable_Name"]
                ),
                "source_description": clean_text(record["Description"]),
                "field_type": clean_text(record["Field_Type"]),
            }
        )

    return (
        "Normalize every metadata record below. Return one JSON object with an "
        "`items` list. The items must remain in the exact same order as the "
        "input records. Each item must contain exactly: standard_name, "
        "domain_category, description, unit_type.\n\n"
        f"Records:\n{json.dumps(items, indent=2)}"
    )


def call_gemini_batch(
    batch_prompt: str,
    model_name: str,
) -> list[NormalizedMetadata]:
    """Send one batch-normalization request to Gemini."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    _throttle()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=batch_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=BatchNormalizedMetadata,
        ),
    )

    if response.parsed is not None:
        parsed = response.parsed
        if isinstance(parsed, BatchNormalizedMetadata):
            return parsed.items

        return BatchNormalizedMetadata.model_validate(parsed).items

    return BatchNormalizedMetadata.model_validate_json(response.text).items


def call_gemini_batch_with_fallback(
    batch_prompt: str,
) -> tuple[list[NormalizedMetadata], str]:
    """Try Gemini models until one successfully normalizes the batch."""
    global _last_gemini_model

    override = os.environ.get("GEMINI_MODEL")
    chain = [override] if override else GEMINI_MODEL_CHAIN
    last_error = None

    for model_name in chain:
        try:
            results = call_gemini_batch(batch_prompt, model_name)
        except Exception as exc:
            if not any(
                code in str(exc)
                for code in ("503", "404", "429", "UNAVAILABLE", "NOT_FOUND")
            ):
                raise

            print(f"  {model_name} unavailable, trying next model...")
            last_error = exc
            continue

        if model_name != _last_gemini_model:
            print(f"  Using Gemini model: {model_name}")
            _last_gemini_model = model_name

        return results, model_name

    raise RuntimeError("All Gemini models in the chain failed.") from last_error


def normalize_metadata_batch(
    records: list[dict],
    llm_provider: str = "gemini",
) -> list[tuple[NormalizedMetadata, str, bool]]:
    """
    Normalize a list of input rows.

    Gemini uses batched requests. OpenAI and offline mode continue using the
    existing one-record-at-a-time normalization behavior.
    """
    if llm_provider != "gemini":
        return [
            normalize_metadata(
                state=record["State"],
                variable_name=record["Variable_Name"],
                description=record["Description"],
                field_type=record["Field_Type"],
                use_llm=(llm_provider != "offline"),
                llm_provider=llm_provider,
            )
            for record in records
        ]

    cache = load_cache()
    results = [None] * len(records)
    uncached_positions = []

    # Reuse existing Gemini cache entries first.
    for position, record in enumerate(records):
        cache_key = build_cache_key(
            record["State"],
            record["Variable_Name"],
            record["Description"],
            record["Field_Type"],
            "gemini",
            "auto",
        )

        if cache_key in cache:
            cached = cache[cache_key]
            results[position] = (
                NormalizedMetadata.model_validate(cached["normalized"]),
                cached["normalization_source"],
                True,
            )
        else:
            uncached_positions.append(position)

    # Send only uncached rows to Gemini, in batches.
    for start in range(0, len(uncached_positions), GEMINI_BATCH_SIZE):
        positions = uncached_positions[start : start + GEMINI_BATCH_SIZE]
        batch_records = [records[position] for position in positions]

        print(
            f"  Normalizing Gemini batch "
            f"{start // GEMINI_BATCH_SIZE + 1} "
            f"({len(batch_records)} variables)"
        )

        batch_prompt = build_batch_user_prompt(batch_records)
        normalized_batch, used_model = call_gemini_batch_with_fallback(batch_prompt)

        if len(normalized_batch) != len(batch_records):
            raise RuntimeError(
                "Gemini returned a different number of normalized records than "
                f"requested: expected {len(batch_records)}, got {len(normalized_batch)}."
            )

        for position, record, normalized in zip(
            positions,
            batch_records,
            normalized_batch,
        ):
            cache_key = build_cache_key(
                record["State"],
                record["Variable_Name"],
                record["Description"],
                record["Field_Type"],
                "gemini",
                "auto",
            )

            cache[cache_key] = {
                "normalization_source": "gemini",
                "model": used_model,
                "normalized": normalized.model_dump(),
            }

            results[position] = (normalized, "gemini", False)

        # Save after every completed batch, so interrupted runs still resume.
        save_cache(cache)

    return results

def call_openai(user_prompt: str, model_name: str) -> NormalizedMetadata:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    # Optional: only needed for a custom gateway. None = standard OpenAI.
    base_url = os.environ.get("OPENAI_BASE_URL") or None

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=NormalizedMetadata,
    )

    normalized = response.choices[0].message.parsed
    if normalized is None:
        raise RuntimeError("OpenAI returned no structured normalization.")
    return normalized


def normalize_metadata(
    state: str,
    variable_name: str,
    description: str,
    field_type: str,
    use_llm: bool = True,
    llm_provider: str = "gemini",
) -> tuple[NormalizedMetadata, str, bool]:
    """
    Normalize one state metadata field.

    Returns (normalized_metadata, normalization_source, cache_hit).
    normalization_source is "gemini", "openai", or "offline_fallback".
    """
    if use_llm and llm_provider not in VALID_PROVIDERS:
        raise ValueError(
            f"llm_provider must be one of {VALID_PROVIDERS}, got {llm_provider!r}"
        )

    provider = llm_provider if use_llm else "offline"

    # Gemini uses "auto" in the cache key because the fallback chain may pick
    # different models mid-run. The model that answered is stored in the record.
    if not use_llm:
        model_name = ""
    elif provider == "gemini":
        model_name = "auto"
    else:
        model_name = get_model_name(provider)

    cache_key = build_cache_key(
        state, variable_name, description, field_type, provider, model_name
    )
    cache = load_cache()

    if cache_key in cache:
        record = cache[cache_key]
        return (
            NormalizedMetadata.model_validate(record["normalized"]),
            record["normalization_source"],
            True,
        )

    if not use_llm:
        normalized = offline_normalization(variable_name, description, field_type)
        source = "offline_fallback"
        used_model = ""
    else:
        user_prompt = build_user_prompt(state, variable_name, description, field_type)
        if provider == "gemini":
            normalized, used_model = call_gemini_with_fallback(user_prompt)
        else:
            normalized = call_openai(user_prompt, model_name)
            used_model = model_name
        source = provider

    cache[cache_key] = {
        "normalization_source": source,
        "model": used_model,
        "normalized": normalized.model_dump(),
    }
    save_cache(cache)

    return normalized, source, False