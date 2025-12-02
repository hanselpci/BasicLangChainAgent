from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


def send_email(to: str, subject: str, body: str):
    """Send an email"""
    email = {"to": to, "subject": subject, "body": body}
    # ... email sending logic

    return f"Email sent to {to} with {email}"


def init_model():
    return init_chat_model(
        model_provider="ollama",
        model="qwen3:0.6b",
        base_url="http://localhost:11434",
        temperature=0.1,
        timeout=30,
        max_tokens=2048,
    )


model = init_model()
agent = create_agent(
    model,
    tools=[send_email],
    system_prompt="You are an email assistant. Always use the send_email tool.",
)
