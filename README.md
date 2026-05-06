# Custom Agent Forge

> An experimental mode-aware knowledge compilation and dataset-forging pipeline for building specialized AI agents from code, documents, notebooks, and reasoning traces.

> Reviewer note: I am a self-taught career changer and this is an experimental learning project. The repository is intended to show systems thinking, documentation, data-pipeline design, and AI workflow experimentation rather than production-level ML engineering.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#requirements)
[![Status](https://img.shields.io/badge/status-active%20research%20prototype-orange.svg)](#project-status)
[![Pipeline](https://img.shields.io/badge/pipeline-DAG%20runtime-informational.svg)](#architecture-overview)
[![Training](https://img.shields.io/badge/training-QLoRA%20%2F%20Unsloth-success.svg)](#training)

**Custom Agent Forge** is the repository for the **Aletheia Knowledge Compiler Engine**: a deterministic data-curation, governance, and model-training pipeline designed to convert heterogeneous source material into high-signal training datasets for specialized AI agents.

The system is intentionally **pipeline-agnostic**. The core compiler does not care what domain the input data comes from; each mode tells the pipeline what structure to expect, how to validate it, and what behavior the resulting agent should learn.

---

## Table of contents

- [Project status](#project-status)
- [What this project does](#what-this-project-does)
- [Operating modes](#operating-modes)
- [Architecture overview](#architecture-overview)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Preparing inputs](#preparing-inputs)
- [Running the orchestrator](#running-the-orchestrator)
- [Dataset generation workflow](#dataset-generation-workflow)
- [Quality gates and governance](#quality-gates-and-governance)
- [Training](#training)
- [API mode](#api-mode)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Safety and reproducibility notes](#safety-and-reproducibility-notes)

---

## Project status

Custom Agent Forge is an active research and engineering project. It already contains the core pieces for ingestion, semantic slicing, DAG scoring, dataset formatting, telemetry, and local LoRA training handoff. Several advanced features are under active development, especially around branch-aware theorist datasets, Semantic Identity Engineering, provenance tracking, and stronger runtime governance.

The current system should be treated as a **research-grade agent-forging stack**, not a turnkey production ML platform. Generated datasets should be inspected before training, and trained adapters should be evaluated against adversarial and out-of-distribution prompts before use.

---

## What this project does

Custom Agent Forge converts raw technical material into structured training data for specialized reasoning agents.

At a high level, it can:

- ingest GitHub repositories, documents, YAML files, and notebook traces;
- slice code and text into structured semantic units;
- route records through mode-specific extraction paths;
- score nodes with DAG, ACS, SIE, topology, and telemetry signals;
- reject malformed, low-value, or behaviorally unsafe samples;
- format accepted samples into ChatML / JSONL training datasets;
- preserve rejected traces for audit and possible preference-data generation;
- hand validated datasets to Unsloth-based QLoRA training workflows.

The goal is not simply to produce text completions. The goal is to compile source material into **behavior-shaping examples** for agents with controlled reasoning patterns.

---

## Operating modes

The engine supports four major specialization modes. These modes define the expected input shape and target behavior, while the underlying pipeline remains general.

| Mode | Purpose | Typical inputs | Target training behavior |
|---|---|---|---|
| `theorist` | Learn from documentation and theoretical text | PDFs, OCR output, markdown, long-form documents | Explain concepts, extract constraints, reason over theoretical frameworks |
| `coding_assistant` | Learn from repositories and implementation patterns | GitHub links, source files, code manifests | Produce code-aware implementation guidance and QLoRA flashcards |
| `advocate` | Act as an architectural reviewer and orchestrator | Design docs, YAML configs, code/context bundles | Identify risks, contradictions, missing constraints, and hallucination vectors |
| `veteran` | Learn from past human-AI debugging loops | Jupyter notebooks, Colab sessions, Copilot traces, failures and fixes | Recognize friction points and reconstruct recovery paths from failed attempts |

A `hybrid` path is also available for workflows that combine advocate-style critique with veteran-style debugging trajectories.

---

## Architecture overview

The repository implements a staged compiler/runtime architecture:

```text
Source material
    |
    v
Landing pad / manifest selection
    |
    v
Mode-specific extraction
    |
    v
Semantic slicing and knowledge-complex generation
    |
    v
DAG scoring, ACS checks, SIE projection, and telemetry
    |
    v
Dataset formatting and behavior gates
    |
    v
QLoRA-ready ChatML / JSONL dataset
    |
    v
Optional Unsloth training and GGUF export
```

The key design principle is separation between **internal governance metadata** and **visible training content**. Runtime scores, identity state, drift signals, and topology metrics may guide whether a sample survives, but the final training rows should expose only the reasoning and outcomes the model is meant to learn.

---

## Repository layout

```text
.
├── config/
│   └── landing_pad/              # Mode-specific input drop zones
├── output/                       # Generated datasets, logs, audits, and knowledge complexes
├── runs/                         # Run-isolated orchestrator artifacts
├── scripts/
│   ├── autonomous_ingestor.py     # GitHub/repo ingestion and rolling-buffer processing
│   ├── bridge_theorist.py         # Theorist bridge from validated nodes to JSONL
│   ├── ral_amplifier.py           # Reasoning amplification layer
│   ├── flashcard_auditor.py       # Dataset and flashcard audit utilities
│   ├── train_adapters.py          # Adapter training helper
│   ├── train_unsloth_local.py     # Local Unsloth QLoRA/GGUF training script
│   └── ...
├── src/
│   ├── api.py                     # FastAPI gateway
│   ├── celery_app.py              # Celery application setup
│   ├── core/                      # Pydantic models and core contracts
│   ├── interfaces/                # Graph and enrichment interfaces
│   ├── pipeline/                  # Orchestrator, DAG runtime, scoring, formatting, compilers
│   ├── tasks/                     # Celery task workflows
│   └── validation/                # Semantic firewall and validation logic
└── tests/                         # Formatter, runtime, telemetry, governance, and integration tests
```

Important entry points include:

| File | Role |
|---|---|
| `src/pipeline/Agent_Forge_orchestrator.py` | Main interactive and resumable pipeline controller |
| `scripts/autonomous_ingestor.py` | Repository ingestion, staging, backlog handling, and compilation |
| `src/pipeline/semantic_slicer_AG.py` | AST-aware and document-aware semantic slicing |
| `src/pipeline/dag_runtime.py` | Runtime authority layer for node lifecycle, scoring, rejection, and reroll |
| `src/pipeline/dag_scoring_pass.py` | DAG scoring and payload wrapping pass |
| `src/pipeline/dataset_formatter.py` | General dataset formatter for multimode outputs |
| `src/pipeline/sft_formatter.py` | Theorist/SFT ChatML formatter |
| `scripts/ral_amplifier.py` | Reasoning amplification and branch-schema generation path |
| `scripts/train_unsloth_local.py` | Local QLoRA training and GGUF conversion path |
| `src/api.py` | FastAPI ingestion and scoring surface |

---

## Installation

### Requirements

- Python 3.10 or newer is recommended.
- Git must be available on the system path for repository ingestion.
- CUDA is recommended for local training, but non-training pipeline stages are designed to run on CPU.
- Redis is required for Celery-backed API/distributed execution workflows.

### Clone the repository

```bash
git clone https://github.com/Jake36999/Custom_Agent_Forge.git
cd Custom_Agent_Forge
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

The dependency stack includes Pydantic, PyYAML, NetworkX, FastAPI, Celery, Redis, Tree-sitter grammars, pandas, jsonlines, tiktoken, SQLGlot, Black, pytest, and related ML/NLP tooling.

---

## Preparing inputs

Place source files into the relevant landing-pad directory:

```text
config/landing_pad/coding_assistant/
config/landing_pad/theorist/
config/landing_pad/advocate/
config/landing_pad/veteran/
```

### Coding assistant mode

Add a `.txt` manifest containing one GitHub repository URL per line:

```text
https://github.com/example/project-one
https://github.com/example/project-two
```

### Theorist mode

Add document-style source material, such as PDFs, OCR text, markdown, or theory notes, to:

```text
config/landing_pad/theorist/
```

### Advocate mode

Add architecture notes, YAML configs, implementation plans, or review bundles to:

```text
config/landing_pad/advocate/
```

### Veteran mode

Add notebooks or session traces containing attempts, errors, diagnostics, and fixes to:

```text
config/landing_pad/veteran/
```

---

## Running the orchestrator

Start the interactive orchestrator:

```bash
python src/pipeline/Agent_Forge_orchestrator.py
```

The orchestrator will scan the appropriate landing-pad directory, let you select targets, create a run directory, and execute the selected mode.

Each run is isolated under:

```text
runs/<run_id>/
```

Typical artifacts include:

```text
runs/<run_id>/targets.txt
runs/<run_id>/knowledge_complexes/
runs/<run_id>/qlora_<mode>_dataset.jsonl
runs/<run_id>/manifest.json
runs/<run_id>/pipeline.log
```

The manifest records run status, generated artifacts, and quality metrics so interrupted or failed runs can be diagnosed and resumed.

---

## Dataset generation workflow

### Coding assistant path

```text
GitHub links
    -> autonomous_ingestor.py
    -> semantic_slicer_AG.py
    -> cognitive_processor.py
    -> dag_scoring_pass.py
    -> dataset_formatter.py
    -> qlora_coding_assistant_dataset.jsonl
```

### Theorist path

```text
Documents / OCR material
    -> semantic_compiler.py
    -> bridge_theorist.py
    -> ral_amplifier.py
    -> sft_formatter.py
    -> qlora_theorist_dataset.jsonl
```

### Veteran path

```text
Notebook / Colab traces
    -> jupyter_trace_ingestor.py
    -> veteran_state_machine.py
    -> veteran_extraction_runner.py
    -> dataset_formatter.py
    -> qlora_veteran_dataset.jsonl
```

### Advocate path

```text
Architecture/config material
    -> advocate_extraction_runner.py
    -> ACS / DAG / telemetry scoring
    -> dataset_formatter.py
    -> qlora_advocate_dataset.jsonl
```

---

## Quality gates and governance

The pipeline contains several defensive layers intended to prevent malformed or unsafe samples from entering training datasets.

### Current guardrails

- schema recognition for Alpaca-style and ChatML-style records;
- JSONL contract validation after formatter output;
- behavioral rejection for empty or underspecified records;
- parrot-failure detection where assistant output copies source/input text;
- semantic firewall checks for forbidden operations and internal override attempts;
- DAG scoring through ACS, SIE, topology, validation, and telemetry signals;
- run manifests and pipeline logs for reproducibility;
- failure-matrix and rejected-trace capture for downstream auditing.

### Important dataset-risk lessons

The project history identified several failure modes that the pipeline now treats as first-class risks:

- **Parrot bug**: assistant output must never be a direct copy of `code_snippet`, OCR source text, or raw input.
- **OCR hallucination**: corrupted symbols, units, and fragmentary concepts can cause confident but false explanations.
- **Template overfitting**: repeated answer skeletons can make a model look reasoned while learning shallow format compliance.
- **Pseudo-branching**: multiple branches must represent distinct mechanisms, not paraphrases of the same path.
- **Leaky evaluation**: train/eval splits must be checked for normalized prompt and concept overlap.

---

## Training

The repository includes a local Unsloth training entry point:

```bash
python scripts/train_unsloth_local.py \
  --dataset runs/<run_id>/qlora_<mode>_dataset.jsonl \
  --output output/models/<model_name>
```

The script is configured for:

- `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`;
- ChatML dataset formatting;
- LoRA rank `r=16`;
- 4-bit loading;
- gradient checkpointing;
- low-memory Windows workarounds;
- optional GGUF export.

Resume training from the latest checkpoint:

```bash
python scripts/train_unsloth_local.py \
  --dataset runs/<run_id>/qlora_<mode>_dataset.jsonl \
  --output output/models/<model_name> \
  --resume
```

Convert an existing checkpoint only:

```bash
python scripts/train_unsloth_local.py \
  --dataset runs/<run_id>/qlora_<mode>_dataset.jsonl \
  --output output/models/<model_name> \
  --gguf-only \
  --checkpoint output/lora_checkpoints/<model_name>/checkpoint-XYZ
```

Before training, inspect the dataset and confirm that the relevant quality gates passed. For branch-reasoning datasets, do not rely on `<think>` coverage alone; branch depth, branch diversity, evidence grounding, parrot similarity, and failure-boundary quality should also be audited.

---

## API mode

The repository also exposes a FastAPI gateway for triggering ingestion and scoring workflows.

Start the API:

```bash
uvicorn src.api:app --reload
```

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/ingest` | Trigger full pipeline execution for a GitHub repository URL |
| `POST` | `/score` | Score pre-partitioned payloads without cloning a repository |
| `GET` | `/status/{run_id}` | Poll Celery workflow status |
| `GET` | `/health` | Check Redis and worker health |

Example:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://github.com/example/project.git"}'
```

---

## Testing

Run the test suite with:

```bash
pytest
```

Relevant test areas include:

- dataset formatting;
- SFT/ChatML schema behavior;
- DAG runtime transitions;
- consistency and contradiction handling;
- telemetry and drift policy;
- adversarial-lens evaluation;
- Celery task routing;
- pathological runtime cases;
- mode-balance weighting;
- scale and stress behavior.

Recommended pre-training checks:

```bash
pytest tests/test_sft_formatter_phase_e.py
pytest tests/test_sft_telemetry_gaps.py
pytest tests/test_refactored_modules.py
pytest tests/test_pathological.py
pytest tests/test_consistency_runtime_integration.py
```

---

## Roadmap

The current roadmap is focused on turning the compiler from a strong data-generation stack into a more explicit governed reasoning runtime.

### Phase 1: Parrot hardening

- Remove or quarantine any path where raw OCR or `code_snippet` text can become assistant output.
- Add strict rejection rules for exact-match or substring copying.
- Preserve source text only as user-side context or hidden audit metadata.

### Phase 2: Branch schema contracts

- Add strict Pydantic contracts for branch-aware theorist outputs.
- Require explicit branch objects with condition, mechanism, effects, failure boundary, risk, and evidence references.
- Keep schema versioning so legacy rows can be isolated or migrated safely.

### Phase 3: Branch evaluator and formatter

- Render branch traces into deterministic ChatML `<think>` blocks.
- Add metrics such as branch count, mechanism diversity, condition coverage, failure-boundary coverage, and parrot similarity.
- Emit `branch_quality_summary.json` or an equivalent audit artifact.

### Phase 4: OCR provenance and uncertainty

- Carry source document, span, page, bounding box, OCR confidence, and symbol-normalization metadata where available.
- Reject or downgrade high-uncertainty source rows instead of forcing fluent explanations.

### Phase 5: Controlled regeneration

- Regenerate a small representative subset before any full corpus rebuild.
- Compare old linear traces against branch-schema traces.
- Audit for usefulness, grounding, branch diversity, and hallucination risk.

### Phase 6: LoRA A/B evaluation

- Train small controlled adapters only after dataset gates pass.
- Evaluate branch reasoning, abstention behavior, failure-boundary specificity, and out-of-distribution robustness.
- Avoid treating format compliance as proof of reasoning quality.

### Phase 7: SIE self-inference and lens expansion

- Extend identity, role, task, thought/action, and lens nodes as typed runtime scaffolding.
- Keep governance metadata separate from visible training targets.
- Promote topology bridges only when evidence-backed rather than lexical-only.

---

## Safety and reproducibility notes

- Keep raw source, generated datasets, manifests, logs, and training outputs versioned or archived by run ID.
- Never train directly on newly generated data without inspecting a sample and reviewing quality metrics.
- Treat rejected records as valuable diagnostic material, not as disposable noise.
- Do not expose hidden runtime scores, identity scaffolding, or internal governance keys as target text unless intentionally training a governance agent.
- For theorist datasets, prefer abstention or uncertainty labeling over forced explanations of corrupted OCR fragments.
- For coding datasets, confirm that generated assistant outputs are not merely copied source snippets unless the training objective explicitly calls for code reproduction.

---

## Minimal quickstart

```bash
git clone https://github.com/Jake36999/Custom_Agent_Forge.git
cd Custom_Agent_Forge

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
python src/pipeline/Agent_Forge_orchestrator.py
```

After a successful run, the primary training artifact will be located at:

```text
runs/<run_id>/qlora_<mode>_dataset.jsonl
```

---

## License

No license file is documented here yet. Add a `LICENSE` file before publishing, redistributing, or accepting external contributions.
