import os
import sys
import asyncio
from pathlib import Path
from dataclasses import dataclass, field

import pytest
import litellm
from google.adk.runners import InMemoryRunner
from google.genai import types

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import (
    GEval,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from deepeval.models.base_model import DeepEvalBaseLLM

# Ensure REPO_ROOT is in path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onboarding.supervisor.agent import root_agent

# ── Agent Execution Tracer ───────────────────────────────────────────────────

@dataclass
class AgentTrace:
    final_response: str
    tools_called: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    trace_dict: dict = field(default_factory=dict)


async def _run_agent_async(input_text: str) -> AgentTrace:
    app_name = "onboarding_evals"
    runner = InMemoryRunner(agent=root_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id="eval_user"
    )

    trace = AgentTrace(final_response="")
    trace.steps.append(f"User: {input_text}")

    trace_dict = {
        "name": "supervisor_agent",
        "type": "agent",
        "input": input_text,
        "output": "",
        "children": []
    }

    async for event in runner.run_async(
        user_id="eval_user",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=input_text)]),
    ):
        for fc in event.get_function_calls():
            args = dict(fc.args or {})
            trace.tools_called.append(fc.name)
            trace.steps.append(f"Tool call → {fc.name}({args})")
            trace_dict["children"].append({
                "name": fc.name,
                "type": "tool",
                "input": args,
                "children": []
            })

        if event.get_function_responses():
            for fr in event.get_function_responses():
                trace.steps.append(f"Tool result ← {fr.name}: {fr.response}")
                for child in reversed(trace_dict["children"]):
                    if child["name"] == fr.name and child.get("output") is None:
                        child["output"] = fr.response
                        break

        if event.content:
            for part in event.content.parts:
                if part.text:
                    if event.is_final_response():
                        trace.final_response = part.text
                        trace.steps.append(f"Final response: {part.text}")
                        trace_dict["output"] = part.text
                    else:
                        trace.steps.append(f"Thought: {part.text}")
                        trace_dict["children"].append({
                            "name": "Thought",
                            "type": "llm",
                            "input": {"prompt": "Reasoning"},
                            "output": part.text,
                            "children": []
                        })

    trace.trace_dict = trace_dict
    return trace


def run_agent(input_text: str) -> AgentTrace:
    """Run the supervisor agent synchronously and return its execution trace."""
    return asyncio.run(_run_agent_async(input_text))


def _build_test_case(golden: Golden) -> tuple[LLMTestCase, AgentTrace]:
    """Run the agent for a Golden test case and return a DeepEval LLMTestCase."""
    trace = run_agent(golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=trace.final_response,
        expected_output=golden.expected_output,
        tools_called=[ToolCall(name=name) for name in trace.tools_called],
        expected_tools=golden.expected_tools,
    )
    test_case._trace_dict = trace.trace_dict
    return test_case, trace


def _trace_as_text(trace: AgentTrace) -> str:
    """Format trace steps as a numbered list of strings."""
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(trace.steps))


# ── DeepEval LLM Judges ───────────────────────────────────────────────────────

class _LiteLLMJudge(DeepEvalBaseLLM):
    def __init__(self, model: str) -> None:
        self.model = model

    def load_model(self) -> "_LiteLLMJudge":
        return self

    def generate(self, prompt: str) -> str:
        resp = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        resp = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def get_model_name(self) -> str:
        return self.model


class _GenAIJudge(DeepEvalBaseLLM):
    def __init__(self, model: str) -> None:
        self.model = model

    def load_model(self) -> "_GenAIJudge":
        return self

    def _get_client(self):
        from google import genai
        return genai.Client(vertexai=True)

    def generate(self, prompt: str) -> str:
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        response = await self._get_client().aio.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text

    def get_model_name(self) -> str:
        return self.model


def get_judge_model() -> DeepEvalBaseLLM:
    """Return a DeepEval-compatible judge matching the MODEL_PROVIDER config."""
    provider = os.getenv("MODEL_PROVIDER", "gemini").lower()

    if provider == "groq":
        return _LiteLLMJudge(os.getenv("GROQ_MODEL", "groq/qwen/qwen3-32b"))

    if provider == "vertex":
        return _GenAIJudge(os.getenv("VERTEX_MODEL", "gemini-2.5-flash-lite"))

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    if not model_name.startswith("gemini/"):
        model_name = "gemini/" + model_name
    return _LiteLLMJudge(model_name)


# ── Golden Dataset Definition ─────────────────────────────────────────────────

golden_dataset = [
    Golden(
        input="¿Cuántos días de vacaciones al año tengo permitidos?",
        expected_output="Tienes derecho a 25 días laborables de vacaciones pagadas al año.",
        expected_tools=[ToolCall(name="agente_rrhh")]
    ),
    Golden(
        input="¿Cómo puedo solicitar un monitor nuevo para mi puesto?",
        expected_output="El agente de IT registrará un ticket de soporte para solicitar el hardware.",
        expected_tools=[ToolCall(name="agente_it")]
    ),
]


# ── Evaluation Tests ─────────────────────────────────────────────────────────

def test_hr_task_completion():
    """Verify that the HR sub-agent correctly answers vacation policies."""
    test_case, _ = _build_test_case(golden_dataset[0])
    metric = TaskCompletionMetric(threshold=0.7, model=get_judge_model())
    assert_test(test_case, [metric])


def test_it_routing_and_tool_correctness():
    """Verify that technical queries route to the IT agent."""
    test_case, _ = _build_test_case(golden_dataset[1])
    metric = ToolCorrectnessMetric(threshold=0.8, model=get_judge_model())
    assert_test(test_case, [metric])


def test_step_efficiency_and_plan():
    """Verify that the supervisor agent plans and executes efficiently without loops."""
    test_case, trace = _build_test_case(golden_dataset[0])
    trace_case = LLMTestCase(
        input=test_case.input,
        actual_output=_trace_as_text(trace),
        expected_output=test_case.expected_output,
        tools_called=test_case.tools_called,
        expected_tools=test_case.expected_tools,
    )
    trace_case._trace_dict = trace.trace_dict

    judge = get_judge_model()
    metric_efficiency = StepEfficiencyMetric(threshold=0.7, model=judge)
    metric_plan = PlanQualityMetric(threshold=0.5, model=judge)

    assert_test(trace_case, [metric_efficiency, metric_plan])
