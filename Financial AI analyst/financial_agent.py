from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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

# Combined Multi-Agent
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    model=groq_model,  # ✅ This ensures Groq is used at the top level too
    instructions=['Always include the sources', 'Use tables to display the data'],
    show_tools_calls=True,
    markdown=True
)

# Run the agent with a financial query
multi_ai_agent.print_response("summarize analyst recommendation and share the latest news for NVDA", stream=True)