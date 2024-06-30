import os
import warnings
from datetime import datetime

from tools import ticker_price
from tools.calculator_tools import CalculatorTools
from tools.ticker_price import TickerPrice

warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import yfinance as yf
from datetime import datetime, timedelta

def get_stock_close_price(ticker, date):
    """Useful to get the close price of a stock at a specific date.
    The input to this tool should be: ticker, date in the format `YYYY-MM-DD`.
    """
    # Convert date string to datetime object and adjust the end date
    date = datetime.strptime(date, '%Y-%m-%d')
    next_day = date + timedelta(days=2)  # Adding two days to ensure the date is included

    # Fetch data
    stock = yf.Ticker(ticker)
    hist = stock.history(start=date.strftime('%Y-%m-%d'), end=next_day.strftime('%Y-%m-%d'))

    # Retrieve close price if available
    try:
        close_price = hist.loc[date.strftime('%Y-%m-%d'), 'Close']
        return close_price
    except KeyError:
        return "No data available for this date."


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
# companies = ["AAPL", "MSFT", "NVDA", "JNJ", "TMO", "MDT", "JPM", "V", "MA", "PG", "PEP", "EL", "XOM", "CVX", "EOG"]
companies = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "NVDA",  # NVIDIA Corporation
    "UNH",  # UnitedHealth Group Incorporated
    "LLY",  # Eli Lilly and Company
    "JNJ",  # Johnson & Johnson
    "HON",  # Honeywell International Inc.
    "UNP",  # Union Pacific Corporation
    "MMM",  # 3M Company
    "TSLA",  # Tesla, Inc.
    "AMZN",  # Amazon.com, Inc.
    "HD",  # The Home Depot, Inc.
    "LIN",  # Linde plc
    "SHW",  # Sherwin-Williams Company
    "FCX"  # Freeport-McMoRan Inc.
]
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

index = 0
total = 0
print("Below are the details of trading the selected companies: \n")
for company in companies:
    index += 1
    buying_price = get_stock_close_price(ticker=company, date="2022-01-03")
    no_stocks = int(inputs["initial_investment"]) / len(companies) / buying_price
    selling_price = get_stock_close_price(ticker=company, date="2023-12-29")
    final_value = selling_price * no_stocks
    total += final_value
    print(
        f"{index}. **{company}**\n-Buying Price: {buying_price}\n-Number of Stocks: {no_stocks}\n-Selling Price: {selling_price}\n-Final Value: {final_value}\n")
print(f"Total Portfolio Value at the end of period: ${total}")
# results = crew.kickoff(inputs=inputs)
#
# print("######################")
# print(results)
