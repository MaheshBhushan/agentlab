# Optional SDK verification

AgentLab keeps SDK imports inside adapters. The `baseline` adapter makes no model
calls. Missing dependencies or provider failures raise explicit errors. No
framework failure selects the baseline automatically.

Install extras independently or together with `uv sync --extra langchain --extra
autogen --extra crewai --extra nemo`. The verified Python version is 3.12.13. The
package supports Python 3.12–3.13 because NeMo 0.24.1 excludes Python 3.14.
The SDKs read `OPENAI_API_KEY` from the environment. The default model is
`gpt-4o-mini`. No credentials ship in the package.

## Execution boundary

`run_adapter(name, request, model="gpt-4o-mini")` returns an `AdapterResult`
containing a strict normalized `TaskOutput` and nullable input/output token
counts. Every adapter receives the same task messages. LangChain adds its agent
system prompt; AutoGen uses its agent system message; CrewAI wraps the shared
instruction in its role, goal and task prompts. These wrappers can affect model
behavior. JSON parsing failures remain failed runs. Provider counts remain null
when the SDK does not report them.

LangChain calls `create_agent` with no tools and a recursion limit. AutoGen calls
`AssistantAgent.run` with an `OpenAIChatCompletionClient` and closes the client
in `finally`. CrewAI calls `Crew.kickoff` for one `Agent` and one `Task` with no
tools, delegation, memory or cache. Provider timeouts are 60 seconds. LangChain
and AutoGen disable provider retries; CrewAI disables agent error retries and
limits iterations. SDKs can still perform internal calls.

`check_input(request, engine, model)` and `check_output(request, output, engine,
model)` return explicit allow, review or block decisions with traceable reasons.
The evaluator records these decisions and routes flagged cases to human review.
The `builtin` engine checks bounded injection patterns, schema, verbatim quoted
evidence, owner/date grounding, invoice fields and sensitive-data patterns.
These heuristics do not establish semantic correctness or comprehensive PII
protection. The evaluator separately checks fixture-specific correctness.

The `nemo` engine loads packaged YAML through `RailsConfig.from_path`, constructs
`LLMRails`, and calls `check` with explicit input or output rail types. Its
configured `self check input` and `self check output` flows call an LLM. NeMo
blocks remain blocks; transformed results require review. Errors propagate as
`GuardError`; AgentLab never substitutes a different guard. Model-based rails
require credentials and add model calls. Their token counts are currently
unknown, so guarded live total cost stays unknown.

## Verified versions and evidence

All extras resolved and installed together in one Python 3.12.13 environment on
2026-10-15:

| Distribution | Installed version |
| --- | --- |
| langchain | 1.4.6 |
| langchain-openai | 1.7.2 |
| autogen-agentchat | 0.7.5 |
| autogen-ext | 0.7.5 |
| crewai | 1.15.28 |
| nemoguardrails | 0.24.1 |

`tests/test_integrations.py` uses local fake SDK modules to check wrapper
arguments, common messages, cleanup, error handling, token counts, benign
controls and malicious inputs. These checks need no SDK installation.

`tests/test_sdk_smoke.py` opts into real installed SDK execution with controlled
responses. It executes LangChain's real agent graph with a fake chat model,
AutoGen's real assistant with its replay client, CrewAI's real agent/task/crew
with a controlled `BaseLLM`, and NeMo's real configuration and self-check flows
with a controlled LangChain LLM. NeMo input and output flows both pass with a
`No` response and block with a `Yes` response. A separate construction check uses
placeholder credentials without invoking providers. This evidence validates SDK
plumbing. It does not validate provider responses or model-based rail quality.

Run installed-SDK checks after installing all extras:

```sh
AGENTLAB_SDK_SMOKE=1 CREWAI_TELEMETRY_DISABLED=true OTEL_SDK_DISABLED=true \
  uv run --all-extras python -m unittest discover -s tests -p test_sdk_smoke.py -v
```

No authenticated provider requests were made during verification. Framework
quality, live token reporting, production security and comparative model results
remain unverified.

## Primary API references

- [LangChain create_agent](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
- [AutoGen agents](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html)
- [AutoGen model clients](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/model-clients.html)
- [CrewAI tasks](https://docs.crewai.com/en/concepts/tasks)
- [CrewAI usage metrics](https://docs.crewai.com/en/concepts/crews)
- [NeMo Python APIs](https://docs.nvidia.com/nemo/guardrails/latest/run-guardrailed-inference/using-python-apis/overview)
- [NeMo LLMRails check API](https://docs.nvidia.com/nemo/guardrails/guardrails-python-sdk/nemoguardrails/rails/llm/llmrails)
