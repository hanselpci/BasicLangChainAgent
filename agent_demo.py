from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


def init_agent():
    model = init_chat_model(
        model_provider="ollama",
        model="qwen3:0.6b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=30,
        max_tokens=2048,
    )

    agent = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
        middleware=[handle_tool_errors],
    )
    return agent


def main():
    agent = init_agent()

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
        stream_mode="values",
    ):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")


# Run the agent by invoke()
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
# )
# print(result)

if __name__ == "__main__":
    main()
