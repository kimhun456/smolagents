import os

from smolagents import OpenAIModel, ToolCallingAgent, tool


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_ID = "gemini-2.5-flash-lite"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


@tool
def count_letters(text: str, letter: str) -> int:
    """
    Count how many times a letter appears in a text.

    Args:
        text: The text to inspect.
        letter: The single letter to count.
    """
    if len(letter) != 1:
        raise ValueError("letter must be exactly one character")

    return text.lower().count(letter.lower())


def get_required_env(name: str) -> str:
    """Return a required environment variable or raise a helpful error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_model() -> OpenAIModel:
    """Create the Gemini-compatible OpenAI model."""
    return OpenAIModel(
        model_id=GEMINI_MODEL_ID,
        api_base=GEMINI_API_BASE,
        api_key=get_required_env(GEMINI_API_KEY_ENV),
    )


def build_agent() -> ToolCallingAgent:
    """Create an agent with the example tools."""
    return ToolCallingAgent(tools=[count_letters], model=build_model())


if __name__ == "__main__":
    agent = build_agent()
    result = agent.run("How many times does the letter 'e' appear in 'Hello there'?")
    print(result)
