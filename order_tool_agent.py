import os

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
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
    return list(range(1, orders_id + 1))


tools = {
    create_order.name: create_order,
    add_item_to_order.name: add_item_to_order,
    remove_item_from_order.name: remove_item_from_order,
    get_order_items.name: get_order_items,
    get_orders.name: get_orders,
}

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in online shop assistent. Your task help consumers",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

llm = ChatMistralAI(
    model="mistral-large-latest",  # pyright: ignore[reportCallIssue]
    temperature=0,
)

agent = create_tool_calling_agent(llm, list(tools.values()), prompt)
agent_executor = AgentExecutor(agent=agent, tools=list(tools.values()), verbose=True)

memory = InMemoryChatMessageHistory()
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output",
)

config = {"configurable": {"session_id": "test-session"}}
while True:
    user_input = input("You: ")
    if user_input.startswith("/bye"):
        break

    result = agent_with_history.invoke({"input": user_input}, config)  # pyright: ignore[reportArgumentType]
    print("Bot:", result["output"])
