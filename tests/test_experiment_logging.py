import json
from unittest.mock import MagicMock

from agent.core.experiment import ExperimentConfig, ExperimentHarness
from agent.core.memory import InMemorySessionMemory
from agent.core.run_logger import JsonlRunLogger
from agent.core.types import NativeModelResponse, RuntimeConfig


def _client(answer="ok"):
    client = MagicMock()
    client.call_with_tools.return_value = NativeModelResponse(
        final_answer=answer,
        tool_calls=[],
        raw_assistant_message={"role": "assistant", "content": answer},
    )
    return client


def test_experiment_harness_logs_each_run(tmp_path):
    path = tmp_path / "experiment.jsonl"
    harness = ExperimentHarness(run_logger=JsonlRunLogger(path))
    config = ExperimentConfig(
        name="memory-a",
        memory_factory=InMemorySessionMemory,
        tools=[],
        model_client=_client(),
        inputs=["one", "two"],
        runtime_config=RuntimeConfig(max_steps=2),
    )

    results = harness.run([config])

    assert len(results["memory-a"]) == 2
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["experiment_name"] for record in records] == ["memory-a", "memory-a"]
    assert [record["input"] for record in records] == ["one", "two"]
    assert [record["final_answer"] for record in records] == ["ok", "ok"]
