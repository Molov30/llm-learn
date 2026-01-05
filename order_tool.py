import json
import os

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.messages.tool import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from pydantic import Field

from settings import settings

SUCCESS_ADD_ITEM_MESSAGE = "success add item message"
ERROR_NO_ORDER_MESSAGE = "error no order message"
SUCCESS_REMOVE_ITEM_MESSAGE = "success remove item message"
ERROR_NO_ITEM_IN_ORDER_MESSAGE = "error no item in order message"

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = settings.api_key

DEFAULT_SESSION_ID = "default"
chat_history = InMemoryChatMessageHistory()
orders_id = 0
orders: list[list[int]] = list()


@tool
def create_order() -> int:
    """Create new order.

    Returns:
        int: Order id
    """
    global orders_id
    orders_id += 1
    orders.append(list())

    return orders_id


@tool
def add_item_to_order(
    order_id: int = Field(description="order id"),
    item_id: int = Field(description="item id"),
) -> str:
    """Add an item to an existing order.

    Args:
        order_id (int): excist order id
        item_id (int): id of the item being added to add id

    Returns:
        str: "success add item message" if successfull operation else "error no order message"
    """
    if (order_id - 1) > len(orders):
        return ERROR_NO_ORDER_MESSAGE

    orders[order_id - 1].append(item_id)
    return SUCCESS_ADD_ITEM_MESSAGE


@tool
def remove_item_from_order(
    order_id: int = Field(description="order id"),
    item_id: int = Field(description="item id"),
) -> str:
    """Remove item in order.

    Args:
        order_id (int): excist order id
        item_id (int): remove item id

    Returns:
        str: "success remove item message" if successfull operation else if order no exists "error no order message" else "error no item in order message"
    """
    if (order_id - 1) > len(orders):
        return ERROR_NO_ORDER_MESSAGE
    order = orders[order_id - 1]
    if item_id not in order:
        return ERROR_NO_ITEM_IN_ORDER_MESSAGE

    order.remove(item_id)
    return SUCCESS_REMOVE_ITEM_MESSAGE


@tool
def get_order_items(order_id: int = Field(description="order id")) -> list[int]:
    """Get order items.

    Args:
        order_id (int): order_id

    Returns:
        list[int]: order items id
    """
    if (order_id - 1) > len(orders):
        return []

    return orders[order_id - 1]


@tool
def get_orders() -> list[int]:
    """Get all orders

    Returns:
        list[int]: all exists orders id
    """
    if len(orders) == 0:
        return []
    return list(range(1, orders_id))


tools = {
    create_order.name: create_order,
    add_item_to_order.name: add_item_to_order,
    remove_item_from_order.name: remove_item_from_order,
    get_order_items.name: get_order_items,
    get_orders.name: get_orders,
}

messages = [
    (
        "system",
        "You are an expert in online shop assistent. Your task help consumers",
    ),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
]
prompt = ChatPromptTemplate(messages)

trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=10,
    start_on="human",
    end_on="human",
    include_system=True,
    allow_partial=False,
)


llm = ChatMistralAI(
    model="mistral-large-latest",  # pyright: ignore[reportCallIssue]
    temperature=0,
)
llm_with_tools = llm.bind_tools(
    [
        create_order,
        add_item_to_order,
        remove_item_from_order,
        get_order_items,
        get_orders,
    ]
)

chain = prompt | trimmer | llm_with_tools

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="history",
)

while True:
    print()
    user_question = input("You: ")
    if user_question.startswith("/bye"):
        break

    print("Bot: ", end="")
    ai_msg = chain_with_history.invoke(
        {"question": user_question},
        config={"configurable": {"session_id": DEFAULT_SESSION_ID}},
    )
    call_tool = False
    for tool_call in ai_msg.tool_calls:
        call_tool = True
        if tool_call["name"] in tools:
            tool_result = tools[tool_call["name"]].invoke(tool_call["args"])
            tool_message = ToolMessage(
                content=json.dumps(tool_result), tool_call_id=tool_call["id"]
            )
            chat_history.add_message(tool_message)

    if call_tool:
        msg = llm_with_tools.invoke(chat_history.messages)
        print(msg.content, end="")
        print()
        continue

    for answer_chunk in chain_with_history.stream(
        {"question": user_question},
        config={"configurable": {"session_id": DEFAULT_SESSION_ID}},
    ):
        print(answer_chunk.content, end="")
    print()
