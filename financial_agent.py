from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

web_search_agent = Agent(
    name="Web Search Agent",
    role = 'search the web for the information',
    model = Groq(id = "llama3-groq-70b-8192-tool-use-preview")
    tools = [DuckDuckGo()],
    instructions = ['always include sources']
    show_tools_calls = True,
    markdown = True
    )