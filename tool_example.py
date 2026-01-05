import os

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from pydantic import Field

from settings import settings

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = settings.api_key

DEFAULT_SESSION_ID = "default"
chat_history = InMemoryChatMessageHistory()


@tool
def solve_equation(
    a: float = Field(description="a coefficient"),
    b: float = Field(description="bias"),
) -> float:
    """
    Equation example: 4*x - 12 = 0
        >>> resutl = solve_equation(4, -12)
        >>> print(result)
        3.0
    """
    return -b / a


llm = ChatMistralAI(
    model="mistral-large-latest",  # pyright: ignore[reportCallIssue]
    temperature=0,
)
llm_with_tools = llm.bind_tools([solve_equation])


ai_message = llm_with_tools.invoke("Solve equal 20x + 100 = 0")
for tool_call in ai_message.tool_calls:
    if tool_call["name"] == solve_equation.name:
        tool_message = solve_equation.invoke(tool_call)
        result = llm_with_tools.invoke(
            [
                HumanMessage("Solve equal 20x + 100 = 0"),
                ai_message,
                tool_message,
            ]
        )
        print(result.content)
