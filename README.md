# AgentLab

AgentLab compares agent frameworks on three construction-office tasks: invoice triage, document questions, and meeting follow-up. The shared input and output contracts keep adapters comparable. All fixture documents are synthetic German office records.

Python 3.12 and uv are required. The core uses the Python standard library.

```sh
uv sync
uv run agentlab list
uv run agentlab run invoice_complete
uv run python -m unittest discover -s tests
```

Expected fixture labels belong to evaluation only. `build_request` copies an allowlist of input fields into framework prompts. Evidence includes a document ID and a verbatim quote. Invoice decisions support human triage and never authorize payment.

The corpus contains 12 cases across normal, missing-data, conflicting-data, ambiguous follow-up, and prompt-injection scenarios. `baseline` uses deterministic line-oriented rules. It provides an offline reference result; it does not measure LangChain, AutoGen, CrewAI, or NeMo. Documents use explicit German field labels, so the baseline cannot interpret unrestricted office prose.
