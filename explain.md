# P1 — Hybrid LLM Jailbreak + Prompt Injection Detector
### Plain-English Build Log

---

## Day 0 — Project Scaffold

**What was built:**
The skeleton of the entire project — folder structure, configuration file, coding standards, and CI quality gates — before any real logic was written.

**Why it matters:**
Starting with a clean scaffold means every future day builds on a solid, consistent foundation. The quality gates (auto-formatting, type-checking, security scan, test coverage) are enforced from day one so bad habits never sneak in.

**How it fits the system:**
Think of this as laying the foundation of a building. The blueprint (`config/config.yaml`) defines every parameter the system will ever use — model names, thresholds, file paths — so nothing is ever hardcoded later.

**Key decisions:**
- Python 3.10 only — stable, widely supported, no experimental features.
- All 6 quality gates run before every commit: black → isort → flake8 → mypy → bandit → pytest.
- The full pipeline architecture was stubbed out: Normalize → Perplexity Gate → FAISS → Stage A → Stage B → Policy Gate.

---

## Day 1 — Security Baseline + Structured Logging + Config Pipeline

**What was built:**
The infrastructure layer — structured JSON logging, config loading from YAML, API request/response schemas, and all the security and exception scaffolding.

**Why it matters:**
Every production ML system needs observable, traceable logs and a single source of truth for configuration. Without this, debugging is impossible and config values scatter across the codebase.

**How it fits the system:**
Every future component (data pipeline, model, API) will use `get_logger()` and `load_config()`. The `ClassifyRequest` / `ClassifyResponse` schemas define the exact contract the API exposes — these never change without a version bump.

**Key decisions:**
- Structured JSON logs (python-json-logger) so logs are parseable by tools like Grafana, Datadog, or CloudWatch.
- Pydantic v2 for API schemas — validates inputs at the boundary, rejects bad data before it reaches the model.
- All exceptions inherit from `ProjectBaseError` — one catch-all for monitoring.

---

## Day 2 — Data Assembly from 5 Sources

**What was built:**
A data pipeline that pulls labeled examples from 5 real-world datasets, validates them against a strict schema, splits them into train/val/test sets, and pre-builds a similarity index of known attacks.

**Why it matters:**
A jailbreak detector is only as good as the variety of attacks it has seen. Using 5 different sources covers a wide range of attack styles: direct jailbreaks, harmful instructions, indirect prompt injections, and safe benign text.

**How it fits the system:**
- The `train.parquet`, `val.parquet`, `test.parquet` files feed directly into the Stage A model training on Day 5.
- The FAISS index (`attack_index.faiss`) is loaded at inference time by the Similarity Gate — it flags prompts that closely resemble known attacks before the model even runs.

**Data sources:**
| Source | Label | Type |
|--------|-------|------|
| JailbreakBench (dedeswim/JBB-Behaviors) | Jailbreak | Direct user prompts |
| AdvBench | Jailbreak | Harmful instructions |
| deepset/prompt-injections | Indirect injection | Injected into retrieved docs |
| BIPIA (Microsoft) | Indirect injection | Email / web context attacks |
| allenai/c4 | Safe | Normal web text |

**Key decisions:**
- Every collector has a try/except — if a dataset is gated or offline, the pipeline degrades gracefully instead of crashing.
- `is_multiturn = False` for all rows: all source datasets are single-turn. At inference, multi-turn conversations are handled by concatenating the history and classifying the full chain as one string.
- Pandera schema enforced on every load — catches dtype mismatches, empty texts, and invalid labels before they corrupt training.
- FAISS `IndexFlatIP` with L2-normalised embeddings = cosine similarity with no approximation error (dataset is small enough for exact search).

**Numbers:**
- 500 total samples (200 per source, capped for dev run)
- Label split: 200 safe · 100 jailbreak · 200 indirect injection
- FAISS index: 210 attack vectors, 384 dimensions
- Test coverage: 85% across 49 tests

---
