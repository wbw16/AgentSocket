from agent import Agent, JsonlRunLogger, ToolSpec, from_env
from agent.memory_backends.simple_demo import SimpleDemoMemory


def echo(args: dict) -> str:
    """Echo工具的具体处理程序，原样返还特定文字。"""
    # return f"echo:{args['text']}"
    return f"Hello, AgentSocket!"


# 组装描述传递给 Agent 的 Echo 工具规范
echo_tool = ToolSpec(
    name="echo",
    description="Echo text",   # 工具说明，用于引发模型兴趣触发
    handler=echo,              # 背后真正的 Python 方法映射
    parameters_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)

# 组织实例化并跑通基于原生的 Agent
agent = Agent(
    model_client=from_env(),                         # 通过提取当前环境量读取配置的 Provider Client
    tools=[echo_tool],                               # 加入了我们构造测试跑用的 `echo` 工具
    memory=SimpleDemoMemory(),                       # 装填了一个用于本地演示和日志快照存档支持的基础版记忆类
    run_logger=JsonlRunLogger("runs/smoke.jsonl"),   # 把单次会话执行日志及调用足迹保留下来
)

# 传入指令开始跑模型并索取最终给用户的明文输出
result = agent.run("请调用 echo 工具，text 参数填 hello world,结果只返回echo工具输出的内容")
print(result.final_answer)
