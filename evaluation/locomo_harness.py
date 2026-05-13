from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Ensure AgentSocket dir is in sys.path so agent.* shim resolves
_AGENT_SOCKET_DIR = Path(__file__).parent.parent
if str(_AGENT_SOCKET_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_SOCKET_DIR))

from agent.core.engine_native import NativeToolEngine
from agent.core.middleware import MiddlewareChain
from agent.core.tools import ToolRegistry
from agent.core.types import MemoryBackend, RuntimeConfig

# Use absolute package imports (AgentSocket is on sys.path via parent dir)
from AgentSocket.datasets.locomo import LoCoMoSample
from AgentSocket.evaluation.metrics import score_result


@dataclass
class LoCoMoConfig:
    name: str
    memory_factory: Callable[[], MemoryBackend]
    model_client: object
    samples: list[LoCoMoSample]
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    sample_ratio: float = 1.0
    max_qa_per_sample: int | None = None  # None = no limit; set to 2 for smoke
    system_prompt: str = (
        "You are a helpful assistant with access to conversation history. "
        "Answer questions based on the conversation context provided. "
        "Be concise and direct."
    )


@dataclass
class QAResult:
    sample_id: str
    question: str
    prediction: str
    ground_truth: str
    category: int | None
    scores: dict


class LoCoMoHarness:
    def run(self, config: LoCoMoConfig) -> list[QAResult]:
        samples = config.samples
        if config.sample_ratio < 1.0:
            n = max(1, int(len(samples) * config.sample_ratio))
            samples = samples[:n]

        results: list[QAResult] = []

        for sample in samples:
            # Each sample gets its own independent memory instance
            memory = config.memory_factory()

            # Phase 1: ingest conversation history
            for session_id in sorted(sample.conversation.sessions.keys()):
                session = sample.conversation.sessions[session_id]
                for turn in session.turns:
                    content = f"[{session.date_time}] {turn.text}"
                    memory.append_message(turn.speaker, content)

            # Phase 2: answer QAs sharing the same memory instance
            qa_list = sample.qa
            if config.max_qa_per_sample is not None:
                qa_list = qa_list[: config.max_qa_per_sample]

            for qa in qa_list:
                if qa.final_answer is None:
                    continue

                engine = NativeToolEngine(
                    model_client=config.model_client,
                    tool_registry=ToolRegistry(tools=[]),
                    memory=memory,
                    middleware_chain=MiddlewareChain(middlewares=[]),
                    runtime_config=config.runtime_config,
                    system_prompt=config.system_prompt,
                )

                run_result = engine.run(qa.question)
                prediction = run_result.final_answer or ""
                ground_truth = str(qa.final_answer) if qa.final_answer is not None else ""

                results.append(
                    QAResult(
                        sample_id=sample.sample_id,
                        question=qa.question,
                        prediction=prediction,
                        ground_truth=ground_truth,
                        category=qa.category,
                        scores=score_result(prediction, ground_truth),
                    )
                )

        return results

    def aggregate(self, results: list[QAResult]) -> dict:
        if not results:
            return {}
        avg_f1 = sum(r.scores["token_f1"] for r in results) / len(results)
        avg_em = sum(r.scores["exact_match"] for r in results) / len(results)
        return {
            "num_questions": len(results),
            "avg_token_f1": avg_f1,
            "avg_exact_match": avg_em,
        }
