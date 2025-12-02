from langchain_ollama import ChatOllama  # old way
from langchain.chat_models import init_chat_model  # new in langchain 1.0


def init_mode_in_old_way():
    model = ChatOllama(
        model="qwen3:0.6b", base_url="http://localhost:11434", temperature=0.1
    )
    return model


def init_model():
    model = init_chat_model(
        model_provider="ollama",
        model="qwen3:0.6b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=30,
        max_tokens=2048,
    )
    return model


def main():
    # model = init_mode_in_old_way()
    model = init_model()
    for chunk in model.stream("写一首唐诗"):
        print(chunk.content, end="", flush=True)


if __name__ == "__main__":
    main()
