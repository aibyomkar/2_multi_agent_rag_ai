import openai
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app
import phi.api
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")

# Initialize Groq model (LLaMA2-70B)
groq_model = Groq(id="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

web_search_agent = Agent(
    name="Web Search Agent",
    role='Search the web for the information',
    model=groq_model,
    tools=[DuckDuckGo()],
    instructions=['Always include sources'],
    show_tools_calls=True,
    markdown=True
)

# Financial Data Agent
finance_agent = Agent(
    name="Finance AI Agent",
    role='Gather financial data',
    model=groq_model,
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=['Use tables to display the data'],
    show_tools_calls=True,
    markdown=True
)

app = Playground([finance_agent, web_search_agent]).get_app()

if __name__ == '__main__':
    serve_playground_app('playground:app', reload=True)