import os
from datetime import datetime
from typing import Annotated, Sequence, TypedDict

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.constants import END, START
from langgraph.graph.message import add_messages
from langgraph.graph.state import StateGraph
from pydantic import BaseModel, Field

from settings import settings


class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], add_messages]
    number_of_steps: int


@tool(return_direct=True)
def get_this_year_tool() -> int:
    """Return current year

    Returns:
        int: current year
    """
    return datetime.now().year


class WikiInput(BaseModel):
    query: str = Field(description="Запрос для поиска в Википедия.")


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="ru"))  # pyright: ignore[reportCallIssue]


@tool(return_direct=True, args_schema=WikiInput)
def search_using_wikipedia(query: str) -> str:
    """Search on wikipedia

    Returns:
        str: search result
    """
    return wikipedia.run({"query": query})


tools = [search_using_wikipedia, get_this_year_tool]
tools_by_name = {tool.name: tool for tool in tools}


def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
    if "MISTRAL_API_KEY" not in os.environ:
        os.environ["MISTRAL_API_KEY"] = settings.api_key

    model = ChatMistralAI(
        model="mistral-large-latest",  # pyright: ignore[reportCallIssue]
        temperature=0,
    )
    model = model.bind_tools(tools)
    response = model.invoke(state["messages"], config)
    return {"messages": [response], "number_of_steps": state["number_of_steps"] + 1}


def call_tool(state: AgentState) -> AgentState:
    if not isinstance(state["messages"][-1], AIMessage):
        return state

    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs, "number_of_steps": state["number_of_steps"] + 1}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
        return "end"
    return "continue"


builder = StateGraph(AgentState)

builder.add_node("llm", call_model)
builder.add_node("tools", call_tool)

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": END})
builder.add_edge("tools", "llm")


graph = builder.compile()
graph.get_graph().draw_png("graph.png")


inputs: AgentState = {
    "messages": [
        HumanMessage(content="Сколько лет у власти последний президент Венесуэлы?"),
    ],
    "number_of_steps": 0,
}
state = graph.invoke(inputs)

for message in state["messages"]:
    message.pretty_print()
    print("=" * 80 + "\n\n")
