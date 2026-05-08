import os

from smolagents import OpenAIModel, ToolCallingAgent, WebSearchTool


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_ID = "gemini-2.5-flash-lite"
MAX_STEPS = 10
PLANNING_INTERVAL = 3


def get_required_env(name: str) -> str:
    """Return a required environment variable or raise a helpful error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_agent() -> ToolCallingAgent:
    """Create a web-search agent backed by Gemini's OpenAI-compatible API."""
    model = OpenAIModel(
        model_id=GEMINI_MODEL_ID,
        api_base=GEMINI_API_BASE,
        api_key=get_required_env("GEMINI_API_KEY"),
    )
    return ToolCallingAgent(
        tools=[WebSearchTool()],
        model=model,
        planning_interval=PLANNING_INTERVAL,
    )


def run() -> None:
    agent = build_agent()
    query = input("Query: ").strip()
    if not query:
        print("No query provided.")
        return

    result = agent.run(task=query, max_steps=MAX_STEPS)
    print(result)


if __name__ == "__main__":
    run()
