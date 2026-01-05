import os

from langchain_classic.agents import Tool
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from settings import settings

if "TAVILY_API_KEY" not in os.environ:
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key


tool = TavilySearchResults(
    max_results=10,
    search_depth="advanced",
    include_answer=False,
    include_raw_content=True,
    include_images=False,
)

result = tool.invoke({"query": "The best course for development MVP AI service"})
print(result)


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="en"))  # pyright: ignore[reportCallIssue]
wikipedia_tool = Tool(
    name="wikipedia",
    description="Search in Wikipedia knowledge database.",
    func=wikipedia.run,
)
result = wikipedia_tool.invoke("Large Language Models")
print(result)
