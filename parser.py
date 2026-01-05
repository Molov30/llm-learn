import os

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from settings import settings

if "MISTRAL_API_KEY" not in os.environ:
    os.environ["MISTRAL_API_KEY"] = settings.api_key


class Person(BaseModel):
    firstname: str = Field(description="firstname of hero")
    lastname: str = Field(description="lastname of hero")
    age: int = Field(description="age of hero")


llm = ChatMistralAI(
    model="mistral-small-2506",  # pyright: ignore[reportCallIssue]
    temperature=0,
)

messages = [
    ("system", "Handle the user query.\n{format_instructions}"),
    ("human", "{user_query}"),
]
prompt_template = ChatPromptTemplate(messages)
output_parser = PydanticOutputParser(pydantic_object=Person)

prompt_value = prompt_template.invoke(
    {
        "format_instructions": output_parser.get_format_instructions(),
        "user_query": "Генрих Смит был восемнацдцателетним юношей, мечтающим уехать в город",
    }
)

answer = llm.invoke(prompt_value.to_messages())
print(output_parser.invoke(answer))
