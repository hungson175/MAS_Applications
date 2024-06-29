import os
import warnings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew, Process

MANAGER_MODEL = "gpt-4-turbo"
AGENT_MODEL = "gpt-4-turbo"
# IP_ADDRESS = "http://192.168.0.223"

# IP_ADDRESS  = "http://localhost"
# MODEL_NAME = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
# MODEL_NAME = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
# load_dotenv(find_dotenv())
#
# llm_lm_studio = ChatOpenAI(
#     openai_api_base=IP_ADDRESS + ":1234/v1",
#     openai_api_key="",
#     model_name=MODEL_NAME,
#     temperature=0.2,
# )
# Load the .env file
load_dotenv()

# Access the API keys
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = AGENT_MODEL

from crewai_tools import ScrapeWebsiteTool, SerperDevTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

dea = Agent(
    role="Data Extraction Specialist",
    goal="Retrieve and preprocess financial documents and real-time market data to provide "
         "foundational data for financial analysis.",
    backstory="Employed at a leading investment bank on Wall Street, this agent is equipped "
              "with a CFA (Chartered Financial Analyst) certification and specializes in "
              "automating the extraction and initial processing of financial documents. "
              "It efficiently handles large volumes of 10-K reports and real-time data feeds,"
              " ensuring accuracy and readiness for analysis, critical for high-stakes "
              "financial decision-making.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool]
)

data_extraction_task = Task(
    description=(
        "Fetch and preprocess 10-K reports for the company represented by {stock_selection} "
        "from {from_year} to {to_year}. Note that: add some delay while using scrape tools (3s-5s), "
        "because SEC site will deny you if you are too fast. Retrieve current stock price data, "
        "and extract structured "
        "financial statements and metrics needed for detailed equity analysis, ensuring data is "
        "timely and accurately reflects the fiscal periods for each year within the range. "
        "Support the Equity Research Analyst with precisely prepared datasets to enable "
        "efficient metric computation."
    ),
    expected_output=(
        "Structured datasets containing relevant sections from 10-K reports and real-time stock price "
        "data for {stock_selection}, covering each year from {from_year} to {to_year}. "
        "The dataset will include segmented financial statements, earnings figures, "
        "and up-to-date stock prices for each specified year, ready for detailed financial analysis."
    ),
    agent=dea,
)

era = Agent(
    role="Equity Research Analyst",
    goal="Analyze financial data extracted by the Data Extraction Specialist to compute key "
         "financial metrics and assess company performance.",
    backstory="This agent works at a prestigious investment bank on Wall Street and holds "
              "both a CPA (Certified Public Accountant) and CFA certification. "
              "It applies advanced quantitative analysis methods to assess a company's "
              "financial health and market position. The agent is highly skilled in"
              " interpreting processed data to calculate crucial financial metrics, "
              "providing actionable insights for portfolio managers and traders.",
    verbose=True,
    allow_delegation=True,
    # tools=[search_tool, scrape_tool]
)

equity_analysis_task = Task(
    description=(
        "Analyze preprocessed financial data for {stock_selection} from {from_year} to {to_year}. "
        "Compute metrics such as Revenue Growth, Profit Margins, ROE, Debt-to-Equity Ratio,"
        " P/E Ratio, and Free Cash Flow for each year within the range. Use this historical "
        "data to provide trend analysis and forecast potential financial positions. "
        "Support decision-making processes with accurate and timely financial assessments."
    ),
    expected_output=(
        "A comprehensive report on key financial metrics for {stock_selection} in 2 forms: \n"
        "1. 6 key metrics in .csv \n"
        "2. Text summary of the analysis\n"
        "including year-over-year growth, profitability analysis, return on equity,"
        " leverage ratios, valuation multiples, and liquidity assessments for each "
        "year from {from_year} to {to_year}. Highlight significant trends and provide"
        " a forward-looking perspective based on historical data."
    ),
    agent=era,
)

crew = Crew(
    agents=[dea, era],
    tasks=[data_extraction_task, equity_analysis_task],
    verbose=True,
    manager_llm=ChatOpenAI(model=MANAGER_MODEL, temperature=0.2),
    process=Process.hierarchical
)
inputs = {
    'stock_selection': 'AAPL',
    'from_year': '2022',
    'to_year': '2023',
    # 'news_impact_consideration': True,
}

result = crew.kickoff(inputs=inputs)

