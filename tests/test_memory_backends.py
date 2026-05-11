from agent.core.types import ToolCallRecord
from agent.memory_backends.simple_demo import SimpleDemoMemory


def test_simple_demo_memory_smoke():
    memory = SimpleDemoMemory()
    memory.append_message("user", "first")
    memory.append_message("assistant", "second")
    memory.append_action(ToolCallRecord(tool="echo", parameters={"msg": "hi"}, result_summary="hi"))

    assert memory.messages() == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    assert memory.action_history()[0].tool == "echo"
    assert memory.retrieve("anything", k=1) == [{"role": "assistant", "content": "second"}]
    assert "first" in memory.summarize(max_tokens=100)
    assert memory.snapshot()["message_count"] == 2
