import os

from smolagents import (
    OpenAIModel,
    WebSearchTool,
    ToolCallingAgent
)

web_search_tool = WebSearchTool()
agent = ToolCallingAgent(
    tools=[web_search_tool],
    model=OpenAIModel(
        model_id="gemini-2.5-flash-lite",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
    ),
    planning_interval=3,
)

system_prompt_step = agent.memory.system_prompt
# print("The system prompt given to the agent was:")
# print(system_prompt_step.system_prompt)

def run():
    query = input("Query : ")
    agent.run(task=query, max_steps=10)

run()
