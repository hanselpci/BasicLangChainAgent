from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient


def init_model():
    return init_chat_model(
        model_provider="ollama",
        model="qwen3:0.6b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=30,
        max_tokens=2048,
    )


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",  # Local subprocess communication
                "command": "python",
                # Absolute path to your math_server.py file
                "args": ["math_server.py"],
            },
            "weather": {
                "transport": "streamable_http",  # HTTP-based remote server
                # Ensure you start your weather server on port 8000
                "url": "http://localhost:8000/mcp",
            },
        }
    )
    tools = await client.get_tools()

    agent = create_agent(init_model(), tools)

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
    )

    print("Math Response:", math_response)
    print("Weather Response:", weather_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
