"""Opt-in installed-SDK execution with controlled responses; never live providers.

Run AGENTLAB_SDK_SMOKE=1 python -m unittest discover -s tests -p test_sdk_smoke.py -v
in an environment with all four extras installed.
"""
import os
import unittest
from unittest.mock import patch
from pathlib import Path

from agentlab.adapters import run_adapter
from agentlab.guards import check_input, check_output
from test_integrations import REQUEST, OUTPUT, RAW


@unittest.skipUnless(os.environ.get("AGENTLAB_SDK_SMOKE") == "1", "opt-in installed SDK smoke")
class InstalledSDKSmoke(unittest.TestCase):
    def test_provider_client_construction_without_network(self):
        import asyncio
        import httpx
        from langchain_openai import ChatOpenAI
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from crewai import LLM
        from agentlab.guards import _nemo_rails
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-placeholder"}):
            with httpx.Client() as transport:
                model = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60,
                                   max_retries=0, http_client=transport)
                self.assertEqual(model.model_name, "gpt-4o-mini")
            client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0,
                                               timeout=60, max_retries=0)
            asyncio.run(client.close())
            self.assertIsNotNone(LLM(model="gpt-4o-mini", temperature=0, timeout=60))
            self.assertIsNotNone(_nemo_rails("gpt-4o-mini"))
            _nemo_rails.cache_clear()

    def test_langchain_real_agent_controlled_chat_model(self):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage
        fake = GenericFakeChatModel(messages=iter([AIMessage(content=RAW)]))
        with patch("langchain_openai.ChatOpenAI", return_value=fake):
            self.assertEqual(run_adapter("langchain", REQUEST).output, OUTPUT)

    def test_autogen_real_agent_replay_client(self):
        from autogen_ext.models.replay import ReplayChatCompletionClient
        replay = ReplayChatCompletionClient(chat_completions=[RAW])
        with patch("autogen_ext.models.openai.OpenAIChatCompletionClient", return_value=replay):
            self.assertEqual(run_adapter("autogen", REQUEST).output, OUTPUT)

    def test_crewai_real_agent_task_crew_controlled_llm(self):
        from crewai import BaseLLM
        class ControlledLLM(BaseLLM):
            def __init__(self, **kwargs): super().__init__(model="controlled-offline", temperature=0)
            def call(self, messages, **kwargs): return RAW
        with patch.dict(os.environ, {"CREWAI_TELEMETRY_DISABLED": "true", "OTEL_SDK_DISABLED": "true"}), \
             patch("crewai.LLM", ControlledLLM):
            self.assertEqual(run_adapter("crewai", REQUEST).output, OUTPUT)

    def test_nemo_real_config_and_self_check_rails_controlled_llm(self):
        from nemoguardrails import RailsConfig, LLMRails
        from langchain_core.language_models.fake import FakeListLLM
        config_dir = Path(__file__).parents[1] / "src/agentlab/guard_config"
        for response, blocked in (("No", False), ("Yes", True)):
            rails = LLMRails(RailsConfig.from_path(str(config_dir)), llm=FakeListLLM(responses=[response]))
            with patch("agentlab.guards._nemo_rails", return_value=rails):
                self.assertEqual(check_input(REQUEST, "nemo").action == "block", blocked)
                self.assertEqual(check_output(REQUEST, OUTPUT, "nemo").action == "block", blocked)
