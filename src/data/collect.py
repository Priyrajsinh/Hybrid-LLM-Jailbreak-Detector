"""
Data collection from 5 sources into a unified schema.

NOTE on is_multiturn:
  All rows have is_multiturn=False because every source dataset is single-turn.
  Multi-turn handling at inference time is a *concatenation heuristic* (C33):
  conversation_history + user_prompt are joined and classified as one string.
  The model has not been trained on multi-turn sequences — this is intentional.
"""

import hashlib
import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = [
    "sample_id",
    "text",
    "label",
    "source_dataset",
    "source_type",
    "language",
    "is_multiturn",
]


def _make_sample_id(source: str, index: int) -> str:
    raw = f"{source}_{index}"
    # usedforsecurity=False: MD5 used only as a short, stable row identifier
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[
        :16
    ]  # nosec B324


def _build_row(
    text: str,
    label: int,
    source_dataset: str,
    source_type: str,
    index: int,
    language: str = "en",
) -> dict[str, Any]:
    return {
        "sample_id": _make_sample_id(source_dataset, index),
        "text": text,
        "label": label,
        "source_dataset": source_dataset,
        "source_type": source_type,
        "language": language,
        "is_multiturn": False,  # All sources are single-turn (see module docstring)
    }


def collect_jailbreakbench(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load jailbreak prompts from JailbreakBench (dedeswim/JBB-Behaviors on HF).
    Label=1 (jailbreak).

    NOTE: Loads directly via HuggingFace datasets rather than the jailbreakbench
    library to avoid its broken litellm dependency at import time.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading JailbreakBench dataset (dedeswim/JBB-Behaviors)")
    dataset = load_dataset(
        "dedeswim/JBB-Behaviors", "behaviors", split="harmful"
    )  # nosec B615
    rows = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        text = str(item.get("Goal", "")).strip()
        if not text:
            continue
        rows.append(_build_row(text, 1, "jailbreakbench", "user_only", i))
    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    logger.info("JailbreakBench: %d rows loaded", len(df))
    return df


def collect_advbench_csv(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load AdvBench harmful behaviors from local CSV at data/raw/harmful_behaviors.csv.
    Label=1 (jailbreak). Skips gracefully if file is absent.
    """
    import os

    csv_path = os.path.join("data", "raw", "harmful_behaviors.csv")
    if not os.path.exists(csv_path):
        logger.warning("AdvBench CSV not found at %s — skipping", csv_path)
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df_raw = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []
    for i, row in df_raw.iterrows():
        if max_samples is not None and len(rows) >= max_samples:
            break
        text = str(row.get("goal", "")).strip()
        if not text:
            continue
        rows.append(_build_row(text, 1, "advbench_csv", "user_only", len(rows)))
    result = pd.DataFrame(rows if rows else [], columns=SCHEMA_COLUMNS)
    logger.info("AdvBench CSV: %d rows loaded", len(result))
    return result


def collect_wildjailbreak(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load jailbreak prompts from allenai/wildjailbreak (public HF dataset, streamed).
    Label=1 (jailbreak). Default cap: 500 rows.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    cap = max_samples if max_samples is not None else 500
    logger.info("Loading allenai/wildjailbreak (cap=%d)", cap)

    def _try_load() -> "Any":  # type: ignore[return]
        for kwargs in [
            {"name": "vanilla", "split": "train", "streaming": True},
            {"split": "train", "streaming": True},
        ]:
            try:
                return load_dataset("allenai/wildjailbreak", **kwargs)  # nosec B615
            except Exception:
                continue
        raise RuntimeError("allenai/wildjailbreak: no loadable config found")

    try:
        ds = _try_load()
        rows: list[dict[str, Any]] = []
        for item in ds:
            if len(rows) >= cap:
                break
            text = ""
            for field in ["vanilla", "forbidden_prompt", "prompt", "text"]:
                text = str(item.get(field, "")).strip()
                if text:
                    break
            if not text:
                continue
            rows.append(_build_row(text, 1, "wildjailbreak", "user_only", len(rows)))
        result = pd.DataFrame(rows if rows else [], columns=SCHEMA_COLUMNS)
        logger.info("WildJailbreak: %d rows loaded", len(result))
        return result
    except Exception as exc:
        logger.warning("WildJailbreak unavailable: %s", exc)
        return pd.DataFrame(columns=SCHEMA_COLUMNS)


def collect_advbench(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load harmful instructions from AdvBench. Label=1 (jailbreak).

    Tries walledai/AdvBench first (may be gated). Falls back to
    Undi95/toxic-dpo-v0.2 which contains similar harmful-behavior prompts.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading AdvBench dataset")

    sources = [
        ("walledai/AdvBench", "train", ["prompt", "text", "instruction"]),
        ("Undi95/toxic-dpo-v0.2", "train", ["prompt", "question", "input"]),
    ]

    for hf_name, split, fields in sources:
        try:
            dataset = load_dataset(hf_name, split=split)  # nosec B615
            rows = []
            for i, item in enumerate(dataset):
                if max_samples is not None and i >= max_samples:
                    break
                text = ""
                for field in fields:
                    text = item.get(field, "") or ""
                    if text:
                        break
                text = str(text).strip()
                if not text:
                    continue
                rows.append(_build_row(text, 1, "advbench", "user_only", i))
            df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
            logger.info("AdvBench (%s): %d rows loaded", hf_name, len(df))
            return df
        except Exception as exc:
            logger.warning("AdvBench source %s unavailable: %s", hf_name, exc)

    logger.error("All AdvBench sources failed — returning empty DataFrame")
    return pd.DataFrame(columns=SCHEMA_COLUMNS)


def collect_prompt_injections(max_samples: int | None = None) -> pd.DataFrame:
    """Load prompt injection examples from deepset/prompt-injections. Label=2."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading deepset/prompt-injections dataset")
    dataset = load_dataset("deepset/prompt-injections", split="train")  # nosec B615
    rows = []
    idx = 0
    for item in dataset:
        if max_samples is not None and idx >= max_samples:
            break
        text = item.get("text") or item.get("prompt") or ""
        text = str(text).strip()
        if not text:
            continue
        # Only use actual injection examples (label=1 in the source)
        src_label = item.get("label", 1)
        if int(src_label) == 0:
            continue
        rows.append(
            _build_row(text, 2, "deepset_prompt_injections", "retrieved_doc", idx)
        )
        idx += 1
    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    logger.info("deepset/prompt-injections: %d rows loaded", len(df))
    return df


def collect_bipia_slices(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load BIPIA (Benchmark for Indirect Prompt Injection Attacks) slices.
    Label=2 (indirect_injection). Loads email and web slices.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading BIPIA benchmark slices")
    rows: list[dict[str, Any]] = []
    slices: list[tuple[str, str]] = [
        ("email", "email"),
        ("web", "web"),
    ]
    for slice_name, source_type in slices:
        if max_samples is not None and len(rows) >= max_samples:
            break
        try:
            ds = load_dataset("microsoft/BIPIA", slice_name, split="test")  # nosec B615
            for i, item in enumerate(ds):
                if max_samples is not None and len(rows) >= max_samples:
                    break
                text = item.get("attack") or item.get("text") or item.get("input") or ""
                text = str(text).strip()
                if not text:
                    continue
                rows.append(
                    _build_row(
                        text,
                        2,
                        f"bipia_{slice_name}",
                        source_type,
                        len(rows),
                    )
                )
            logger.info("BIPIA %s: %d rows loaded", slice_name, i + 1)
        except Exception as exc:
            logger.warning("BIPIA %s slice unavailable: %s", slice_name, exc)
    if not rows:
        # Fallback: HackAPrompt competition dataset (public, prompt injection attempts)
        cap = max_samples if max_samples is not None else 500
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            for split_name in ["train", "test", "validation"]:
                try:
                    ds_hack = load_dataset(  # nosec B615
                        "hackaprompt/hackaprompt-dataset",
                        split=split_name,
                        streaming=True,
                    )
                    for item in ds_hack:
                        if len(rows) >= cap:
                            break
                        if not item.get("injection_successful", False):
                            continue
                        text = ""
                        for field in ["user_input", "prompt", "text", "input"]:
                            text = str(item.get(field, "")).strip()
                            if text:
                                break
                        if not text:
                            continue
                        rows.append(
                            _build_row(
                                text, 2, "hackaprompt", "retrieved_doc", len(rows)
                            )
                        )
                    if rows:
                        break
                except Exception:
                    continue
            logger.info("HackAPrompt fallback: %d rows loaded", len(rows))
        except Exception as exc:
            logger.warning("HackAPrompt unavailable: %s", exc)

    df = pd.DataFrame(rows if rows else [], columns=SCHEMA_COLUMNS)
    logger.info("BIPIA total: %d rows loaded", len(df))
    return df


def collect_safe_corpus(max_samples: int | None = None) -> pd.DataFrame:
    """
    Load safe/benign text examples from allenai/c4 (en, validation split).
    Fallback: stas/openwebtext-10k. Label=0 (safe).
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    logger.info("Loading safe corpus")
    cap = max_samples if max_samples is not None else 15000
    rows: list[dict[str, Any]] = []

    try:
        ds = load_dataset(  # nosec B615
            "allenai/c4",
            "en",
            split="validation",
            streaming=True,
        )
        for i, item in enumerate(ds):
            if len(rows) >= cap:
                break
            text = str(item.get("text", "")).strip()
            if len(text) < 20:
                continue
            # Keep only sentences that are clearly safe (short snippet, no commands)
            snippet = text[:512]
            rows.append(_build_row(snippet, 0, "c4_safe", "web", i))
        logger.info("c4 safe corpus: %d rows loaded", len(rows))
    except Exception as exc:
        logger.warning("c4 unavailable (%s), falling back to openwebtext-10k", exc)
        try:
            ds2 = load_dataset("stas/openwebtext-10k", split="train")  # nosec B615
            for i, item in enumerate(ds2):
                if len(rows) >= cap:
                    break
                text = str(item.get("text", "")).strip()
                if len(text) < 20:
                    continue
                snippet = text[:512]
                rows.append(_build_row(snippet, 0, "openwebtext", "web", i))
            logger.info("openwebtext safe corpus: %d rows loaded", len(rows))
        except Exception as exc2:
            logger.error("Both safe corpus sources failed: %s", exc2)

    df = pd.DataFrame(rows if rows else [], columns=SCHEMA_COLUMNS)
    logger.info("Safe corpus total: %d rows loaded", len(df))
    return df


def collect_all_sources(config: dict[str, Any]) -> pd.DataFrame:
    """
    Merge all 5 sources into a unified DataFrame with the canonical schema.

    Sources collected:
      1. JailbreakBench    — label=1 (jailbreak)
      2. AdvBench          — label=1 (jailbreak)
      3. deepset/prompt-injections — label=2 (indirect_injection)
      4. BIPIA slices      — label=2 (indirect_injection)
      5. c4/OpenWebText    — label=0 (safe)

    is_multiturn is always False. See module docstring for rationale.
    """
    data_cfg = config.get("data", {})
    sources = data_cfg.get("sources", [])
    max_per_source: int | None = data_cfg.get("max_samples_per_source")

    logger.info("Collecting data from sources: %s", sources)

    frames: list[pd.DataFrame] = []

    if "jailbreakbench" in sources:
        frames.append(collect_jailbreakbench(max_per_source))
    if "advbench" in sources:
        frames.append(collect_advbench(max_per_source))
    if "advbench_csv" in sources:
        frames.append(collect_advbench_csv(max_per_source))
    if "wildjailbreak" in sources:
        frames.append(collect_wildjailbreak(max_per_source))
    if "deepset_prompt_injections" in sources:
        frames.append(collect_prompt_injections(max_per_source))
    if "bipia_slices" in sources:
        frames.append(collect_bipia_slices(max_per_source))
    if "safe_corpus" in sources:
        frames.append(collect_safe_corpus(max_per_source))

    if not frames:
        logger.warning("No source frames collected — returning empty DataFrame")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)

    # Coerce dtypes — empty DataFrames from failed sources produce object columns
    combined["label"] = combined["label"].astype(int)
    combined["is_multiturn"] = combined["is_multiturn"].astype(bool)
    combined["sample_id"] = combined["sample_id"].astype(str)
    combined["text"] = combined["text"].astype(str)
    combined["source_dataset"] = combined["source_dataset"].astype(str)
    combined["source_type"] = combined["source_type"].astype(str)
    combined["language"] = combined["language"].astype(str)

    # Deduplicate by text content
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info(
        "Deduplication: %d -> %d rows (%d duplicates removed)",
        before,
        len(combined),
        before - len(combined),
    )

    # Re-assign sample_ids to guarantee uniqueness after concat
    combined["sample_id"] = [
        _make_sample_id(str(row["source_dataset"]), pos)
        for pos, (_, row) in enumerate(combined.iterrows())
    ]

    logger.info(
        "Total rows: %d | Label distribution: %s",
        len(combined),
        combined["label"].value_counts().to_dict(),
    )
    logger.info(
        "Samples per source:\n%s",
        combined["source_dataset"].value_counts().to_string(),
    )

    return combined
