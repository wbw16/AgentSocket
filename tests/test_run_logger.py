import json

from agent.core.memory import InMemorySessionMemory
from agent.core.run_logger import JsonlRunLogger
from agent.core.types import AgentRunResult, ToolCallRecord


def _result(session_id="run-1"):
    return AgentRunResult(
        session_id=session_id,
        final_answer="answer",
        steps=[],
        action_history=[
            ToolCallRecord(tool="lookup", parameters={"q": "x"}, result_summary="found")
        ],
        stop_reason="finished",
        metrics={"duration_seconds": 0.1},
    )


def test_jsonl_run_logger_appends_run_record(tmp_path):
    path = tmp_path / "runs" / "records.jsonl"
    memory = InMemorySessionMemory()
    memory.append_message("user", "hello")

    logger = JsonlRunLogger(path)
    logger.record_run(
        user_input="hello",
        result=_result(),
        memory=memory,
        experiment_name="smoke",
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == "run-1"
    assert record["experiment_name"] == "smoke"
    assert record["input"] == "hello"
    assert record["final_answer"] == "answer"
    assert record["stop_reason"] == "finished"
    assert record["metrics"] == {"duration_seconds": 0.1}
    assert record["action_history"] == [
        {"tool": "lookup", "parameters": {"q": "x"}, "result_summary": "found", "intent": None}
    ]
    assert record["memory_messages"] == [{"role": "user", "content": "hello"}]
    assert "timestamp" in record


def test_jsonl_run_logger_includes_optional_memory_snapshot(tmp_path):
    class SnapshotMemory(InMemorySessionMemory):
        def snapshot(self):
            return {"items": [1, 2, 3]}

    path = tmp_path / "records.jsonl"
    logger = JsonlRunLogger(path)
    logger.record_run(user_input="x", result=_result(), memory=SnapshotMemory())

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["memory_snapshot"] == {"items": [1, 2, 3]}
