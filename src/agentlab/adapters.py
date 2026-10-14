"""Optional framework runners. Importing AgentLab never imports a framework."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any

from .models import TaskOutput
from .tasks import TaskRequest, request_messages

ADAPTERS = ("baseline", "langchain", "autogen", "crewai")


class AdapterError(RuntimeError):
    """Dependency, provider, or normalized-output failure."""


@dataclass(frozen=True)
class AdapterResult:
    output: TaskOutput
    input_tokens: int | None = None
    output_tokens: int | None = None


def _parse(raw: Any, request: TaskRequest) -> TaskOutput:
    if isinstance(raw, str):
        raw = json.loads(raw)
    output = TaskOutput.from_dict(raw)
    if output.task != request.task:
        raise ValueError("adapter returned a different task")
    return output


def _usage(value: Any, key: str) -> int | None:
    count = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
    return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None


def _langchain(request: TaskRequest, model: str) -> AdapterResult:
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    import httpx

    messages = request_messages(request)
    with httpx.Client() as client:
        llm = ChatOpenAI(model=model.removeprefix("openai:"), temperature=0,
                         timeout=60, max_retries=0, http_client=client)
        agent = create_agent(model=llm, tools=[], system_prompt=messages[0]["content"])
        state = agent.invoke({"messages": messages[1:]}, config={"recursion_limit": 10})
    response = state["messages"][-1]
    usage = [getattr(item, "usage_metadata", None) for item in state["messages"]
             if getattr(item, "type", None) == "ai" or hasattr(item, "usage_metadata")]
    # Missing provider counts remain unknown, rather than synthetic zeroes.
    inputs = [_usage(item, "input_tokens") for item in usage]
    outputs = [_usage(item, "output_tokens") for item in usage]
    return AdapterResult(_parse(response.content, request),
                         sum(inputs) if inputs and all(x is not None for x in inputs) else None,
                         sum(outputs) if outputs and all(x is not None for x in outputs) else None)


async def _autogen_async(request: TaskRequest, model: str) -> AdapterResult:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    messages = request_messages(request)
    client = OpenAIChatCompletionClient(model=model.removeprefix("openai:"), temperature=0,
                                       timeout=60, max_retries=0)
    try:
        agent = AssistantAgent("construction_assistant", model_client=client,
                               system_message=messages[0]["content"])
        result = await agent.run(task=messages[1]["content"])
        usage = client.total_usage()
        return AdapterResult(_parse(result.messages[-1].content, request),
                             _usage(usage, "prompt_tokens"), _usage(usage, "completion_tokens"))
    finally:
        await client.close()


def _crewai(request: TaskRequest, model: str) -> AdapterResult:
    from crewai import Agent, Task, Crew, LLM

    messages = request_messages(request)
    llm = LLM(model=model.removeprefix("openai:"), temperature=0, timeout=60)
    agent = Agent(role="Construction office assistant", goal=messages[0]["content"],
                  backstory="You extract verifiable facts from construction office documents.",
                  llm=llm, tools=[], allow_delegation=False, allow_code_execution=False,
                  max_iter=2, max_retry_limit=0, verbose=False)
    task = Task(description=messages[1]["content"], expected_output=messages[0]["content"], agent=agent)
    crew = Crew(agents=[agent], tasks=[task], memory=False, cache=False, verbose=False)
    result = crew.kickoff()
    usage = crew.usage_metrics
    return AdapterResult(_parse(result.raw, request), _usage(usage, "prompt_tokens"),
                         _usage(usage, "completion_tokens"))


def run_adapter(name: str, request: TaskRequest, model: str = "gpt-4o-mini") -> AdapterResult:
    """Run one request. Framework failures never fall back to the offline baseline."""
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter: {name}")
    try:
        if name == "baseline":
            from .baseline import run_baseline
            return AdapterResult(run_baseline(request))
        if name == "langchain":
            return _langchain(request, model)
        if name == "autogen":
            # The CLI is synchronous. Reject nested loops without leaking a coroutine.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_autogen_async(request, model))
            raise AdapterError("autogen synchronous runner requires a thread outside the active event loop")
        return _crewai(request, model)
    except ImportError as exc:
        raise AdapterError(f"{name} dependency unavailable; install agentlab[{name}]") from exc
    except AdapterError:
        raise
    except Exception as exc:
        # Provider messages can contain credentials or request bodies. Preserve cause for
        # local debugging, while keeping the public error safe for JSONL reports.
        raise AdapterError(f"{name} execution failed ({type(exc).__name__})") from exc
