import os
from collections import deque

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI

from settings import settings

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = settings.api_key


llm = ChatMistralAI(
    model="mistral-small-2506",  # pyright: ignore[reportCallIssue]
    temperature=0,
)

messages = [
    (
        "system",
        "You are expert in {domain}. Your task in answer the question as short as possible",
    ),
    MessagesPlaceholder("history"),
]

prompt_template = ChatPromptTemplate(messages)

domain = input("Choise domain area: ")
history = deque(maxlen=10)
while True:
    print()
    user_content = input("You: ")
    if user_content.startswith("/bye"):
        break

    history.append(HumanMessage(content=user_content))
    prompt_value = prompt_template.invoke({"domain": domain, "history": list(history)})
    full_ai_content = ""
    print("Bot: ", end="")
    for ai_message_chunk in llm.stream(prompt_value.to_messages()):
        print(ai_message_chunk.content, end="")
        if isinstance(ai_message_chunk.content, str):
            full_ai_content += ai_message_chunk.content

    history.append(AIMessage(content=full_ai_content))
    print()
