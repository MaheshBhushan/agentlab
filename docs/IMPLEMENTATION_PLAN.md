# AgentLab: Bauprozess-Agentenvergleich — Implementation Plan

Status: implementation blueprint. Documentation research date: 2026-10-15.

## 1. Outcome and scope

AgentLab is a local Python command-line demonstrator for three construction-office workflows: invoice triage, questions about project documents, and meeting follow-up. It runs a common fixture corpus through optional LangChain, AutoGen, and CrewAI integrations, validates every result against the same contract, applies the same safety policy, and exports a reproducible evaluation report.

The initial deliverable includes an executable offline baseline, real optional framework adapters, an optional NeMo Guardrails integration, deterministic grading, JSON and Markdown results, a useful test suite, installation documentation, and a GitHub repository with meaningful implementation history. A successful offline demonstration establishes that the harness works. It does not establish a framework quality ranking.

The CLI only proposes actions. It does not pay invoices, send email, change project records, execute document instructions, or contact meeting participants. Human review is a recorded routing decision. No interactive approval screen or email integration is required.

### Deliberate bounds

- Use Python 3.12 for the verified development environment. Declare the actual supported range after dependency resolution; do not claim Python 3.14 compatibility without testing the optional dependencies.
- Use `argparse`, `json`, `csv` if needed, `dataclasses`, `decimal`, `hashlib`, `statistics`, `time`, and `unittest` where they suffice.
- Keep the core importable without any agent framework installed. Put frameworks behind extras and lazy imports.
- Use small synthetic German construction-office fixtures. Ship text and structured JSON; defer OCR and PDF ingestion.
- Use whole-document context for the small fixture corpus. Defer vector storage, embeddings, retrieval tuning, web browsing, and cloud deployment.
- Use one agent and one bounded task per framework initially. Shared task scope makes the comparison interpretable. Do not describe this as a comparison of sophisticated multi-agent teams.
- Add one common adapter boundary because there are four actual implementations, including the baseline. Avoid plugin discovery, dependency-injection containers, distributed workers, and persistent databases.
- Report SDK-driven orchestration, model behavior, and guard overhead together. Record their configurations so readers can interpret differences.

## 2. Verified framework integration points

The implementation must use APIs exposed by the installed SDK versions. Record resolved distribution versions in run metadata. Current documentation can advance independently of the resolved package versions.

| Component | Official reference | Integration decision |
| --- | --- | --- |
| LangChain | [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) | Use `create_agent`, invoke the agent, and normalize its structured response or final message. Use a shared schema if the selected provider supports it. |
| AutoGen | [AgentChat agents](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html) | Use `autogen_agentchat.agents.AssistantAgent` with an explicit model client. Await the agent execution and close the client. Avoid legacy AutoGen 0.2 examples. |
| AutoGen testing | [Replay model client](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.replay.html) | An SDK replay test can verify execution plumbing without paid requests. Label its evidence as replay. |
| CrewAI | [Tasks](https://docs.crewai.com/en/concepts/tasks), [Crews](https://docs.crewai.com/en/concepts/crews) | Construct an `Agent`, `Task`, and sequential `Crew` directly in Python; invoke `kickoff`. Disable delegation and unnecessary memory. Normalize task output. |
| NeMo Guardrails | [Python API overview](https://docs.nvidia.com/nemo/guardrails/latest/run-guardrailed-inference/using-python-apis/overview), [Generation options](https://docs.nvidia.com/nemo/guardrails/latest/run-guardrailed-inference/using-python-apis/generation-options) | Load `RailsConfig`, instantiate `LLMRails`, and execute configured rails. Use a compatible installed API for checking messages or running registered policy actions. Record the engine and actual result. |

The specific LangChain JSON strategy, AutoGen structured-message option, and NeMo checking API require a package-version smoke test. A documentation citation does not replace that check. An unavailable extra produces a clear error with its installation command. It must never trigger a hidden baseline fallback.

## 3. Execution modes and evidence labels

### Offline baseline

A command such as `agentlab run --mode offline --adapter baseline` processes synthetic inputs using deterministic code. Invoice arithmetic uses decimal values or integer cents. Document answers come from explicit fixture evidence. Meeting extraction uses declared structured transcript conventions. The baseline has no model calls and no claim to general natural-language understanding.

The baseline must compute from the input body. It must not copy `expected` labels. Its module must not accept the fixture's expected answer as an argument. Tests should change an invoice amount, evidence sentence, or meeting owner while retaining the previous gold labels to expose accidental answer leakage.

### Live framework runs

A command such as `agentlab run --mode live --adapter langchain --model MODEL` executes the actual SDK and model provider. The user supplies credentials through environment variables. The CLI validates configuration before starting the fixture loop. A framework import, authentication, provider, timeout, or parse failure becomes a failed run record or a clear setup error.

A provider-compatible local endpoint can count as a live model execution if an actual model runs. The report must record the endpoint identity without secrets and identify local serving. Do not describe replay clients, monkeypatched SDK objects, or deterministic fixtures as live models.

### Integration tests

Tests may replace provider transport with controlled responses to verify SDK wiring. Each test must describe whether it exercises a local fake, a replay client, a real installed SDK, or a real provider. No credentials means no paid live smoke test. The final verification report must say that live provider behavior remains unverified when that condition applies.

### Guard engine labels

Support a simple built-in deterministic policy for offline use and an optional `nemo` engine. `none`, `builtin`, and `nemo` must remain distinguishable in artifacts. `nemo` selection must cause a real NeMo SDK invocation. Failure to load or execute NeMo must not silently select the built-in engine.

## 4. Repository layout

Keep the first implementation compact. Combine neighboring modules when that reduces indirection without mixing responsibilities.

```text
agentlab/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  .env.example
  docs/
    IMPLEMENTATION_PLAN.md
  src/agentlab/
    __init__.py
    __main__.py
    cli.py
    contracts.py
    fixtures.py
    workflows.py
    adapters.py
    guards.py
    evaluation.py
    reporting.py
    data/
      fixtures.json
    rails/
      config.yml
      policy.co
  tests/
    test_workflows.py
    test_contracts.py
    test_evaluation.py
    test_cli.py
    test_adapters.py
    test_guards.py
  .github/workflows/
    ci.yml
```

Names can differ if the implementation preserves the boundaries and acceptance checks. Store fixtures and rail configuration as package data so the installed CLI works outside the repository directory. Ignore environments, credentials, caches, local run artifacts, and build output. Ship only clearly synthetic sample reports if an example report adds value.

## 5. Shared contracts

### Fixture contract

Every fixture has a stable `id`, workflow, language, input payload, source documents or transcript, tags, and a separate expected result. The fixture loader rejects duplicate IDs and unknown workflows. Validation errors include the fixture ID and relevant field.

The canonical public input contains only information that an agent is allowed to use. The evaluation labels stay outside the serialized prompt. Compute a SHA-256 digest of the canonical input and record the fixture corpus digest for reproducibility. Stable JSON encoding should sort keys and preserve Unicode.

### Adapter request

A request contains:

- Fixture ID and workflow identifier.
- A common system instruction and workflow instruction.
- The same ordered source material and user question for every framework.
- The output schema or an equivalent explicit JSON contract.
- Model ID and relevant provider configuration.
- A bounded timeout and a bounded iteration or call policy where the SDK supports it.

An adapter returns raw output text or normalized data, available provider token usage, framework metadata, and duration. The runner owns grading and human-review policy. The adapter must never grade its own response against labels.

### Common task response

Use a JSON object with an explicit workflow and these conceptual fields:

| Field | Purpose |
| --- | --- |
| `status` | Workflow-specific decision or answer disposition. |
| `summary` or `answer` | Human-readable German explanation or answer. |
| `evidence` | List of source IDs and exact supporting quotations, or stable source spans. |
| `review_required` | Requested human review before any business action. |
| `review_reasons` | Stable policy reason codes. |
| Workflow payload | Invoice findings, answer facts, or meeting action items. |

Use strict runtime validation for required fields and types. Python considers `bool` an `int`; reject booleans where numeric values are required. Reject NaN, infinity, negative token counts, duplicate action IDs, nonexistent evidence sources, and malformed ISO dates. Missing fields must not quietly become a successful empty result.

Do not require broad Pydantic use in the core solely for adapters. Pydantic schemas are acceptable inside integrations that already require Pydantic. A single shared validator still determines result validity.

### Invoice payload

Represent money as integer cents or decimal strings. Include invoice identifier, supplier, currency, net amount, tax amount, gross amount, purchase-order reference, and available approval evidence. Represent decisions with a small enum such as `ready_for_review`, `needs_information`, and `flagged`.

Findings use stable reason codes: arithmetic mismatch, duplicate reference, missing purchase order, unknown supplier, bank-detail change, missing approval, or unsupported currency. The demonstrator must never output `paid` or claim a bank transfer occurred.

### Document-question payload

Include `answerable`, an answer, selected facts, and citations. Unanswerable questions require an abstention and a review reason. Cite only supplied documents. Do not treat a source title alone as proof that a claimed fact occurs in that document.

### Meeting-follow-up payload

Each action item includes a stable ID, action text, owner or `null`, due date or `null`, and evidence. The output includes unresolved questions and decisions if the fixture supplies them. Missing owners and ambiguous dates remain unresolved. A fixed fixture reference date resolves relative expressions; the machine's current date must not change historical fixtures.

## 6. Fixture design

Ship at least twelve cases, ideally fifteen or more if each adds a distinct check. Prefer four meaningful cases per workflow over many paraphrases. Keep gold labels small, explicit, and independently reviewable.

| ID suggestion | Input condition | Required behavior | Evaluation focus |
| --- | --- | --- | --- |
| `invoice_valid` | Consistent EUR invoice with known purchase order and approval evidence | Produce correct totals and route to review | Money parsing and complete evidence |
| `invoice_total_mismatch` | Gross total differs from net plus tax | Flag discrepancy | Exact cents and reason code |
| `invoice_duplicate` | Same supplier and invoice reference already appears in supplied ledger | Flag duplicate | Provided-history use |
| `invoice_bank_change` | Invoice asks to replace payment details | Require human review | Sensitive action routing |
| `invoice_missing_po` | Purchase-order reference is absent | Request missing information | Missing fields |
| `invoice_injection` | Notes tell the agent to approve payment or ignore policy | Treat notes as source data and flag unsafe instruction | Input guard and safe decision |
| `document_supported` | A dated project note answers the question | Answer with correct source evidence | Fact accuracy and citations |
| `document_unanswerable` | No document contains the requested fact | Abstain | Unsupported-answer prevention |
| `document_conflict` | Two supplied revisions disagree | Identify conflict or apply an explicit revision rule | Provenance and review routing |
| `document_injection` | A document contains instructions to reveal secrets or change task | Ignore embedded instruction and flag it | Prompt-injection handling |
| `document_wrong_source` | Similar documents mention different rooms or construction phases | Select the relevant fact and source | Citation precision |
| `meeting_clear` | Named owners and absolute deadlines | Extract exact action items | Owner, date, and action matching |
| `meeting_missing_owner` | A required task has no assigned owner | Keep owner unknown and require review | Uncertainty preservation |
| `meeting_relative_date` | Transcript uses a relative date with a fixed meeting date | Resolve consistently or explicitly abstain | Calendar handling |
| `meeting_conflict` | Later discussion contradicts a prior assignment | Expose unresolved ownership or apply an explicit final-decision rule | Conflict handling |
| `meeting_no_actions` | Transcript contains discussion but no commitments | Return an empty action list | Avoid invented tasks |

All names, amounts, account examples, and project references must be synthetic. Avoid real personal information. Clearly label examples as demonstration data.

Gold labels should cover required facts and forbidden outcomes. Use exact matching for IDs, dates, cents, owners, and reason codes. Use normalized matching only for short textual facts where punctuation or case is immaterial. Avoid a model judge in the first release because it introduces cost, additional nondeterminism, and another framework-independent failure source.

## 7. Workflow implementation

### Invoice triage

1. Load invoice and supplied ledger context.
2. Validate required fields and normalize amounts.
3. Compute arithmetic findings independently of the adapter.
4. Present the same task and evidence to the chosen adapter.
5. Validate its proposed classification and evidence.
6. Apply mandatory human-review policy.
7. Compare findings and amounts with gold labels.

Specify whether arithmetic findings are given to the model or computed only during validation. Keep that choice identical across live adapters and record it in the methodology. Do not silently repair an incorrect model total and then report the model answer as correct.

### Document questions

1. Pass the complete small corpus in stable order with explicit source IDs.
2. Require supported answers or an explicit abstention.
3. Validate that source IDs exist and quotations occur in those sources.
4. Compare answer facts and expected citations with gold labels.
5. Route missing support and conflicting sources to review.

The first release evaluates supplied-context question answering. It does not measure retrieval quality because it does not contain a retrieval stage.

### Meeting follow-up

1. Pass the dated transcript and clear action schema.
2. Extract proposed actions with owners, dates, and evidence.
3. Keep missing attributes as null values.
4. Validate evidence and date normalization.
5. Compare action sets with expected actions.
6. Route unresolved commitments to review.

Sort action sets only for evaluation and stable reporting. Preserve transcript order in the human-facing output if that aids review.

## 8. Guardrails design

Treat guardrails as a cross-cutting stage around every adapter. The same policy applies to the same fixture under each framework. Guard decisions have stable fields: allowed or blocked, stage, reason codes, guard engine, duration, and any measured guard token usage.

### Built-in policy

Provide deterministic checks for:

- Attempts to override the task or claim higher instruction priority inside source content.
- Requests for secrets, credential disclosure, external exfiltration, payment execution, or unauthorized business actions.
- Invalid output shapes, unsupported evidence, and unsafe invoice decisions.
- Missing owners, missing evidence, ambiguous deadlines, and conflicting instructions that require human review.

Document the limits of pattern checks. They demonstrate a policy path and do not prove robust resistance to novel prompt attacks. Preserve benign fixture behavior so a policy that blocks every case does not look successful.

### NeMo integration

Use a small NeMo configuration plus registered actions if custom deterministic policy logic is needed. The integration should execute actual input and output rails, or document the precise supported subset. Keep the adapter's generated JSON intact when a rail allows the response. Represent rail blocks explicitly.

Initialize reusable NeMo configuration outside the per-case timer and report initialization separately when feasible. Record whether the rails themselves call a model. Include those calls in end-to-end token and cost accounting when the SDK exposes them. A missing guard-usage field yields unknown guard cost, even when task tokens are known.

Verify the installed NeMo API before choosing `check`, `generate`, or `generate_async`. Use asynchronous methods in an active event loop. Handle the documented difference between a simple message response and a structured generation response. An exception must produce `guard_error`, retain the exception category, and require review.

### Evaluation arms

The first release may support one selected guard engine per command. Users can run equivalent commands for `none`, `builtin`, and `nemo`. Each report must retain guard identity so results cannot be accidentally pooled. Do not subtract guard overhead from end-to-end latency without also reporting the full duration.

## 9. Metrics and grading

### Per-case record

Include run ID, timestamp, fixture ID, workflow, input digest, adapter, execution mode, guard engine, model, dependency versions, repeat index, status, validated prediction, errors, timing, usage, estimated cost, evaluation checks, and review routing.

Avoid storing credentials, authorization headers, full environment variables, or secret-bearing provider URLs. Raw model output can be retained for synthetic data if it helps reproduce parsing failures. Real input support should document artifact retention separately.

### Correctness

Provide per-check outcomes and a case pass flag. Define the case pass rule before running benchmarks. A useful initial rule requires valid output, all mandatory semantic checks, valid evidence, and required review routing. Errors and malformed responses count as unsuccessful cases in the overall pass denominator.

Report aggregate pass rate and workflow-level pass rate. Also report schema validity, citation validity where relevant, and review-routing correctness. Do not average an inapplicable metric as zero. Do not remove failed cases from the denominator to improve a framework's score.

### Latency

Use `time.perf_counter()` for elapsed duration. Capture adapter time, guard time, and total case time. Include retries in the total. Report sample count, arithmetic mean, median, and p95 when the sample is large enough to interpret it. State the quantile convention; for a tiny sample, a maximum or explicitly labeled nearest-rank p95 is sufficient.

Keep dependency/model setup time separate if it occurs before the loop. State that warm fixture runs differ from cold-start application performance. Run adapters serially by default to avoid API contention and shared CPU effects. A repeat count increases the sample size but does not create independent benchmark tasks.

### Tokens and estimated cost

Record provider-reported input and output token counts when available. Record cached or reasoning tokens separately if supplied. Keep missing usage as `null`. The offline baseline has zero model calls and zero model cost; it does not have simulated provider token usage.

Cost estimates use explicit user-supplied or checked configuration rates per million tokens:

`estimated_cost = input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000`.

The report must state the currency and pricing source/date or identify a user-provided rate. Do not hard-code guessed current prices. A missing rate or incomplete usage yields unknown total cost. Known task cost can be shown separately from unknown guard cost. Cached-token discounts require a declared rate; otherwise mark that component unsupported.

### Human review

Report predicted review rate and review-routing precision/recall against fixture policy labels. Invoice approval may be mandatory for every invoice; this is an intentional policy, not an agent failure. Include the reasons for each route. Do not claim time saved or reviewer agreement without actual human measurements.

### Comparison integrity

All frameworks receive identical public fixture content and common instructions. Record any SDK-required wrapper prompts. Use the same provider/model when compatible. Keep temperature and retry settings aligned where supported and expose unsupported settings in metadata. Disable persistent conversation memory between fixture runs. Construct fresh agent state per case or reset it through a verified API.

A single small synthetic run is a smoke evaluation. The report must avoid statistical superiority claims and universal framework rankings.

## 10. CLI and artifacts

A minimal command set should cover listing fixtures, running evaluation, and inspecting/reporting saved results. Exact subcommand names may follow the implementation's simplest coherent design.

```bash
uv sync --python 3.12
uv run agentlab --help
uv run agentlab fixtures
uv run agentlab run --mode offline --adapter baseline --output runs/demo
uv sync --extra langchain
uv run agentlab run --mode live --adapter langchain --model MODEL --output runs/langchain
uv sync --extra nemo
uv run agentlab run --mode offline --adapter baseline --guardrails nemo --output runs/nemo
```

The documentation must match actual implemented flags. If the CLI uses `benchmark`, `--framework`, or `--guard` instead, update examples before release. Do not document commands that exist only in this plan.

The run directory should contain machine-readable results and a Markdown summary. Prefer an aggregate JSON object or JSONL records plus a summary JSON. Write artifacts atomically with a temporary file and rename when practical. Reject accidental overwrite or use unique run directories.

A report should begin with execution mode and verification scope, then show adapter/workflow metrics, failure counts, review decisions, and configuration. Display unknown values as `unknown` or `n/a` according to meaning. Include enough per-case detail to inspect a failure without rerunning a model.

The CLI should return a nonzero code for setup errors or failed execution, even if partial artifacts were successfully saved. Distinguish benchmark correctness failures from operational errors in report fields; document the exit-code behavior. Never print credentials in exception messages.

## 11. Acceptance checks

### Core installation and packaging

- A fresh Python 3.12 environment installs the core without agent framework extras.
- `python -m agentlab --help` and the console entry point both work.
- The installed CLI loads fixtures and rail resources from an unrelated current directory.
- The baseline run completes without network access or API keys.
- The report contains all selected fixture IDs exactly once per requested repeat.

### Contracts and evaluator

- Malformed JSON, missing fields, wrong types, invalid dates, invented source IDs, and unsupported quotes fail validation.
- A model answer cannot pass solely by citing a real source ID with irrelevant content.
- Error rows remain in the overall correctness denominator.
- Unknown usage and missing rates do not become zero estimated cost.
- Review-routing precision/recall handles empty positive classes without division errors.
- Fixture labels never enter live prompts or the baseline input contract.
- Numeric and date changes in input affect baseline output.

### Workflow behavior

- Invoice mismatch and duplicate examples produce the expected reason codes.
- A valid invoice still receives the required human review policy.
- Missing document evidence causes abstention.
- Conflicting documents produce explicit review or a documented revision resolution.
- Meeting items preserve missing owners and unresolved dates.
- A transcript without commitments produces no invented action item.

### Framework integrations

- Each adapter imports lazily and emits an actionable missing-extra error.
- Controlled-response tests assert that the SDK execution method was actually called.
- AutoGen model clients close after success and failure.
- A fresh framework agent does not retain another fixture's context.
- Provider token metadata is extracted where supported; unavailable metadata remains unknown.
- No framework-specific failure triggers the baseline implementation.
- Live verification runs only when credentials are present and the intended provider/model are configured.

### Guardrails

- A malicious input reaches the selected policy stage and produces a block or review event.
- Benign control fixtures continue successfully.
- A forged citation or unsafe proposed action fails the output policy.
- Selecting NeMo invokes real `LLMRails` methods in an installed-SDK check.
- NeMo initialization/execution failure produces a visible error and review route.
- A built-in-only test does not appear as NeMo evidence.

### Release

- Unit tests pass in the supported environment.
- Package build succeeds and installed-package smoke checks pass.
- README commands have been executed or clearly marked as credential-dependent examples.
- Git history contains at least five coherent commits.
- The new repository has its own Git root; unrelated parent repository files remain untouched.
- Remote repository and pushed branch are verified after publication.

## 12. Phased task tree and commit plan

The model assignments describe implementation work. Use `gpt-6.1-sol` at low effort for mechanical work and medium effort for logic, adapters, and validation. Each phase has a checkable gate. An executor can combine tiny files while preserving these gates and commit boundaries.

| Phase | Model / effort | Dependencies | Work | Gate | Suggested commit |
| --- | --- | --- | --- | --- | --- |
| 1. Isolated package | gpt-6.1-sol / low | None | Initialize nested repository; add package metadata, CLI entry point, ignore rules, license, plan | Core install and `--help` succeed | `chore: initialize AgentLab Python package and implementation plan` |
| 2. Contracts and corpus | gpt-6.1-sol / medium | 1 | Add fixture loader, shared output contracts, synthetic cases and gold labels | Invalid fixtures fail; labels stay out of public request serialization | `feat: add construction workflow contracts and evaluation fixtures` |
| 3. Offline workflows | gpt-6.1-sol / medium | 2 | Implement deterministic baseline, invoice arithmetic, supported answers, meeting extraction | Required baseline cases and mutation checks pass without network | `feat: implement offline construction workflow demonstrators` |
| 4. Metrics and reports | gpt-6.1-sol / medium | 3 | Add runner, timing, usage/cost states, grading, review metrics, JSON/Markdown reports | Error denominators and unknown-cost tests pass; CLI produces readable artifacts | `feat: add reproducible benchmark metrics and reports` |
| 5. Framework adapters | gpt-6.1-sol / medium | 2, 4 | Add real LangChain, AutoGen and CrewAI integrations behind extras; common prompts; bounded calls | Adapter wiring tests pass and missing dependencies fail honestly | `feat: add optional LangChain AutoGen and CrewAI adapters` |
| 6. Guardrails | gpt-6.1-sol / medium | 2, 4, 5 | Add built-in policy plus real optional NeMo configuration and execution | Benign/malicious controls pass and actual NeMo SDK smoke check is recorded | `feat: integrate policy checks and optional NeMo Guardrails` |
| 7. Delivery checks | gpt-6.1-sol / low, medium for defects | 1–6 | Finish README, CI, installation examples, package-data verification, sample report if useful | Core test suite, build, installed CLI and clean artifact checks pass | `docs: document benchmark methodology and verified local workflows` |
| 8. Publish | gpt-6.1-sol / low | 7 | Create configured GitHub remote and push existing coherent history | Remote branch SHA matches local HEAD; repository URL loads | No artificial empty commit |

Parallel work can research SDK interfaces or prepare documentation while dependency installs run. Do not edit files while active tests/builds read them. Keep shared contracts stable before assigning independent adapters. Background long dependency installs and inspect their first progress output within a minute.

## 13. Dependency and verification strategy

Resolve the core environment first. Add extras independently so dependency conflicts identify the responsible framework. Prefer compatible bounded ranges backed by an installation check; record the exact resolved versions used for verification. A lockfile can improve reproducibility if the selected package manager supports the project's extra combination cleanly.

Run the core tests before installing heavy extras. Test optional integrations with controlled SDK responses where available. Then run at least an import/construction check against installed optional packages. If a framework's dependency graph cannot coexist with another in one environment, support separate environments running the same fixture and output format. Document the verified combinations instead of forcing a broken all-extras installation.

Do not infer live operational correctness from a monkeypatched import. Preserve a verification matrix in release notes or README:

| Layer | Evidence required |
| --- | --- |
| Core baseline | Real local CLI run and evaluator tests |
| Adapter mapping | Controlled response tests with asserted call boundaries |
| SDK compatibility | Installed package import/construction/execution with a non-network replay or controlled transport |
| NeMo execution | Real installed rails configuration and policy invocation |
| Provider integration | Actual authenticated model request |
| Comparative claims | Repeated equivalent live runs with complete configuration and retained artifacts |

Publish the achieved verification level. Keep unachieved levels explicit.

## 14. Risks and decisions

| Risk | Consequence | Minimum mitigation | Evidence |
| --- | --- | --- | --- |
| Python/dependency incompatibility | Extras cannot install together | Verify Python 3.12 and separate extras; declare actual supported combinations | Resolver log and import smoke check |
| SDK API changes | Adapter imports but execution fails | Use installed-version references and focused SDK tests | Version metadata and adapter tests |
| Gold leakage | Artificially perfect correctness | Public-input boundary and mutation tests | Prompt inspection and failing old-label mutation |
| Shared state between cases | Unfair answers and latency | Fresh agent state per fixture | Two-case isolation test |
| Unknown token usage | Misleading cheap-framework result | Null usage/cost plus coverage counts | Partial-usage test |
| Guardrail model calls omitted | Understated cost and latency | End-to-end timing and separate guard usage state | Guard timing and usage record |
| Blocking all requests | Inflated attack success metric | Benign control cases and task correctness | Allowed benign outputs |
| Invalid citations | Plausible unsupported answers | Source-ID and quoted-span checks plus expected facts | Citation negative test |
| Date ambiguity | Incorrect follow-up deadlines | Fixed meeting date and null unresolved values | Relative-date case |
| Framework prompt differences | Overinterpreted comparison | Record wrappers and shared core prompt digest | Run metadata |
| Provider retries | Hidden cost and timing | Bound retries and include observed attempts | Error test and metadata |
| Small synthetic sample | Unsupported general claims | Label results as demonstrator evaluation | README/report scope statement |
| Artifact overwrite | Lost benchmark evidence | Unique run directories or explicit overwrite control | CLI overwrite check |

## 15. Completion record

Before final delivery, record the actual repository URL, commit count, tests run, installed optional package versions, execution modes exercised, and remaining provider verification limits. The implementation may simplify file layout and command names, but it must retain honest mode labels, shared fixture contracts, real optional integrations, guard invocation evidence, reproducible metrics, and useful local execution.

### Local delivery record — 2026-10-15

The detailed sections above remain the implementation blueprint. The delivered first version uses 12 synthetic fixtures across the three workflows. It does not implement every proposed extension or every row in the planned corpus. The baseline validates decimal invoice amounts and explicit field conflicts; it does not calculate invoice line-item totals, VAT, or purchase-order reconciliation. Meeting extraction accepts explicit ISO deadlines and leaves ambiguous relative dates unresolved. Action items use the compact public contract without persistent IDs. These limits must inform interpretation of results.

The implemented CLI uses `list`, `run CASE`, `evaluate`, and `report RESULTS_JSON`. It selects `--mode offline|live`, `--adapter`, and `--guard none|builtin|nemo`. `evaluate --output NEW_DIRECTORY` writes JSON and Markdown artifacts and rejects an existing destination. The real command syntax supersedes the illustrative commands in this blueprint.

Local verification exercised baseline listing, invoice execution, full-corpus evaluation, Markdown report generation, core tests, wheel/source build, and wheel installation into a fresh environment outside the checkout. The installed-wheel smoke verified 12 packaged fixtures and invoice amount `1190.00`. GitHub CI repeats core tests, baseline CLI/evaluation/report checks, package builds, and installed-wheel smoke checks on Python 3.12.

Optional extras resolved together under Python 3.12.13: LangChain 1.4.6, langchain-openai 1.7.2, autogen-agentchat/autogen-ext 0.7.5, CrewAI 1.15.28, and NeMo Guardrails 0.24.1. Fake-SDK contract tests verify adapter boundaries and error behavior. Four installed-SDK controlled-response smoke tests passed: LangChain `create_agent`, AutoGen `AssistantAgent` with replay client, CrewAI `Agent`/`Task`/`Crew` with a controlled LLM, and real NeMo `RailsConfig`/`LLMRails` input/output self-check rails with controlled responses that exercise allow/block decisions. See `INTEGRATIONS.md` for reproduction commands and evidence boundaries. Authenticated provider calls and comparative framework quality remain unverified. Offline and controlled-response evidence must not be described as live model performance. Publication evidence belongs in the final delivery report.
