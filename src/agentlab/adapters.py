"""Optional framework runners. Importing AgentLab never imports a framework."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .models import TaskOutput
from .tasks import TaskRequest, request_messages

ADAPTERS = ("baseline", "langchain", "autogen", "crewai", "openclaw")
OPENCLAW_TIMEOUT = 60


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


def openclaw_binary() -> str | None:
    return os.environ.get("AGENTLAB_OPENCLAW_BIN") or shutil.which("openclaw")


def openclaw_version() -> str | None:
    binary = openclaw_binary()
    if not binary:
        return None
    try:
        done = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30,
                              env={"PATH": os.environ.get("PATH", ""), "NO_COLOR": "1"})
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() or None


def _openclaw_config(model: str) -> dict:
    """Pinned run config: one OpenAI-compatible model and no model-visible tools."""
    return {
        "models": {"providers": {"agentlab": {
            "baseUrl": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            # OpenClaw resolves the key from the run environment; it never enters argv or the file.
            "apiKey": "${OPENAI_API_KEY}", "api": "openai-completions",
            "models": [{"id": model, "name": model, "reasoning": False, "input": ["text"],
                        "contextWindow": 128000, "maxTokens": 4096}]}}},
        "tools": {"profile": "minimal", "deny": ["*"], "toolSearch": False, "codeMode": False,
                  "exec": {"mode": "deny"}, "fs": {"workspaceOnly": True}},
    }


def _openclaw(request: TaskRequest, model: str) -> AdapterResult:
    binary = openclaw_binary()
    if not binary:
        raise AdapterError("openclaw CLI unavailable; install it with npm install -g openclaw")
    messages = request_messages(request)
    model = model.removeprefix("openai:")
    # agent exec has no system-message input; the shared instruction leads the prompt.
    prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    with tempfile.TemporaryDirectory(prefix="agentlab-openclaw-") as root:
        home, workspace = Path(root, "home"), Path(root, "workspace")
        home.mkdir()
        workspace.mkdir()
        config = Path(root, "openclaw.json")
        config.write_text(json.dumps(_openclaw_config(model)), encoding="utf-8")
        # A fresh HOME hides operator config, plugins and stored logins from the run.
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "NO_COLOR": "1",
               "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")}
        done = subprocess.run(
            [binary, "agent", "exec", "--config", str(config), "--cwd", str(workspace),
             "--model", f"agentlab/{model}", "--thinking", "off",
             "--timeout", str(OPENCLAW_TIMEOUT), "--json", "--message-file", "-"],
            input=prompt, capture_output=True, text=True, env=env, cwd=workspace,
            timeout=OPENCLAW_TIMEOUT + 30)
    try:
        envelope = json.loads(done.stdout)
    except ValueError:
        raise AdapterError(f"openclaw exited {done.returncode} without a JSON result") from None
    if envelope.get("status") != "ok" or not isinstance(envelope.get("final"), str):
        kind = (envelope.get("error") or {}).get("kind") or envelope.get("status")
        raise AdapterError(f"openclaw run failed ({kind})")
    usage = envelope.get("usage") or {}
    return AdapterResult(_parse(envelope["final"], request), _usage(usage, "input"), _usage(usage, "output"))


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
        if name == "crewai":
            return _crewai(request, model)
        return _openclaw(request, model)
    except ImportError as exc:
        raise AdapterError(f"{name} dependency unavailable; install agentlab[{name}]") from exc
    except AdapterError:
        raise
    except Exception as exc:
        # Provider messages can contain credentials or request bodies. Preserve cause for
        # local debugging, while keeping the public error safe for JSONL reports.
        raise AdapterError(f"{name} execution failed ({type(exc).__name__})") from exc
