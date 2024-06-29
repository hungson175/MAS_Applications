import os
import warnings

from tools.calculator_tools import CalculatorTools
from tools.ticker_price import TickerPrice

warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

MANAGER_MODEL = "gpt-3.5-turbo"
AGENT_MODEL = "gpt-4-turbo"

load_dotenv()

# Access the API keys
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = AGENT_MODEL

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

portfolio_value_calculation_agent = Agent(
    role="Portfolio Value Calculation Agent",
    goal="Calculate the total portfolio value by buying stocks on the first trading day of a "
         "given period and selling on the last trading day of the same period for a list of "
         "selected companies.",
    backstory="The agent uses historical stock data to calculate investment returns based on the initial "
              "investment strategy.",
    verbose=True,
    allow_delegation=True,
    max_iter=1000,
    max_execution_time=None,
    tools=[TickerPrice.get_stock_close_price, CalculatorTools.calculate],
)

portfolio_value_calculation_task = Task(
    description=(
        "Calculate the total portfolio value by buying stocks of the selected companies on the "
        "first trading day of the given period ({from_year}) and selling on the last trading day "
        "of the same period ({to_year}). "
        "Use the initial investment amount of {initial_investment}, divided equally among the "
        "selected companies ({list_of_companies})."
    ),
    expected_output=(
        "1. Total portfolio value at the end of the given period.\n"
        "2. Detailed report of the buy and sell prices, number of shares bought, and final "
        "value for each company."
    ),
    agent=portfolio_value_calculation_agent,
)
companies = ["AAPL", "MSFT", "NVDA", "JNJ", "TMO", "MDT", "JPM", "V", "MA", "PG", "PEP", "EL", "XOM", "CVX", "EOG"]

inputs = {
    "from_year": "2022",
    "to_year": "2023",
    "initial_investment": "1000000",
    "list_of_companies": ",".join(companies),
}

crew = Crew(
    agents=[portfolio_value_calculation_agent],
    tasks=[portfolio_value_calculation_task],
    verbose=True,
)

results = crew.kickoff(inputs=inputs)

print("######################")
print(results)
