# AgentLab

AgentLab evaluates three construction-office workflows with a shared synthetic German corpus: invoice triage, questions about project documents, and meeting follow-up. It provides a deterministic offline baseline and optional LangChain, AutoGen, CrewAI, and NVIDIA NeMo Guardrails integrations.

The CLI proposes decisions for human review. It never pays invoices, sends messages, or changes project records. The 12-case corpus exercises normal inputs, missing and conflicting information, ambiguous follow-up, and prompt injection. It supports a reproducible demonstration; it cannot establish a general framework ranking.

## Quick start

Use Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --frozen
uv run agentlab list
uv run agentlab run invoice_complete
uv run agentlab evaluate --output artifacts/offline-demo
uv run agentlab report artifacts/offline-demo/results.json
uv run python -m unittest discover -s tests -v
```

The core requires no framework dependencies or credentials. The baseline reads explicit line-oriented German fields and computes its results from input documents. It cannot interpret unrestricted office prose. Evaluation labels stay outside the public request and framework prompt.

## Workflows

| Workflow | Input | Proposed output |
| --- | --- | --- |
| Invoice triage | Synthetic invoices and review notes | Extracted fields, amount validation, evidence, and review routing |
| Document questions | Question and small whole-document context | Supported answer with document IDs and verbatim quotes, or abstention |
| Meeting follow-up | Structured meeting transcript | Action items, owners, dates, evidence, and unresolved details |

Invoice amount parsing and validation use decimal values. The baseline does not reconcile line items, VAT, or purchase-order totals. Missing information and conflicting sources require review. Meeting deadlines come from explicit ISO date fields; ambiguous relative dates remain unresolved. The harness checks quoted spans against source documents; a matching quote alone does not prove an answer is semantically correct.

## Optional framework runs

Install only the extra you need. Each framework executes a fresh bounded task through its actual SDK. Provider failures produce explicit errors; the runner never substitutes the baseline.

```sh
uv sync --extra langchain
# Or: uv sync --extra autogen
# Or: uv sync --extra crewai
# Or: uv sync --extra nemo
```

A credential-dependent example:

```sh
uv run agentlab evaluate --mode live --adapter langchain --model gpt-4o-mini --output artifacts/langchain-live
```

Use a fresh output directory for each run. Repeat `--adapter` to compare frameworks in one live evaluation, and use `--repetitions N` for repeated attempts. `--guard none|builtin|nemo` selects the guard engine. `--input-price` and `--output-price` accept user-supplied prices per million tokens; `--currency` labels the result. The default run uses `--mode offline --adapter baseline --guard builtin`.

The initial adapters use OpenAI clients and a selected model. Set `OPENAI_API_KEY` in your shell or secret manager. `.env.example` documents the variable; AgentLab does not automatically load `.env` files. Optional SDKs can require network access and incur provider charges. Run metadata records resolved SDK versions.

## Guardrails and evaluation

The `builtin` policy uses deterministic patterns and source checks. It can block suspected input attacks and route incomplete or unsupported output for review. The `none` policy disables these checks. The `nemo` policy invokes the installed NeMo SDK and packaged rail configuration. NeMo is an optional integration and has its own dependency and provider requirements. These policies have bounded detection coverage and require benign controls alongside attack cases.

Metrics describe task correctness, execution failures, evidence validity, latency, review routing, and attack handling. Human review means a routing decision; this project does not measure reviewer labor or final business outcomes. Token usage and cost remain unknown when the provider does not expose complete counts or prices. The offline baseline reports zero monetary model cost and unknown token counts. Live unknown values must not be read as zero cost. Guard execution time belongs to the observed end-to-end latency.

The JSON artifact retains per-attempt records, configuration, versions, and digests. The Markdown report summarizes every attempt, including errors and blocked cases in pass-rate denominators. Exit code 0 means evaluation completed; inspect the report for failed attempts. Setup, input, and artifact errors return exit code 2. Output directories cannot overwrite prior results.

All shipped records are synthetic. Do not infer performance on production construction documents from the offline scores. Repeated live runs with equivalent models, settings, corpus, and retained artifacts are required for a comparative claim.

## Development and verification

```sh
uv build
```

CI runs the core suite and CLI smoke checks on Python 3.12, builds the wheel, and exercises the installed CLI outside the checkout. Installed-SDK smoke checks exercised all four integrations with controlled responses, including real NeMo input/output rails. These checks made no authenticated provider requests. See [integration verification](docs/INTEGRATIONS.md) for versions and reproduction commands. Optional provider tests require credentials. See [the detailed implementation plan](docs/IMPLEMENTATION_PLAN.md) for contracts, methodology, acceptance gates, and the delivery record.

MIT License. Copyright 2026 Koduri Mahesh Bhushan Chowdary.
