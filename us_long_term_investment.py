import os
import warnings

from langchain_openai import ChatOpenAI

from tools.calculator_tools import CalculatorTools

warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

MANAGER_MODEL = "gpt-3.5-turbo"
AGENT_MODEL = "gpt-3.5-turbo"

load_dotenv()

# Access the API keys
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = AGENT_MODEL

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# strategic_investment_system_manager = Agent(
#     role="Strategic Investment System Manager",
#     goal="Oversee the multi-agent system designed to backtest long-term investment strategies. Ensure each agent operates efficiently and collaboratively to identify and analyze top-performing industries and companies based on defined financial metrics.",
#     backstory="This agent holds a Master’s degree in Finance and an MBA, along with CFA and CIMA certifications. "
#               "With a proven track record in investment management and strategic planning, the agent has extensive experience in overseeing data-driven projects and teams within the financial sector. "
#               "The agent is responsible for strategic decision-making, validating outputs, and ensuring the system's alignment with broader investment goals.",
#     verbose=True,
#     allow_delegation=True,
# )
#
# strategic_investment_system_manager_task = Task(
#     description=(
#         "Oversee the multi-agent system for backtesting long-term investment strategies. Coordinate the workflow among"
#         " the Industry Analyzer Agent, Metrics Evaluation Agent, Company Selection Agent, and Top Picks Analysis Agent. "
#         "Validate outputs at each stage, provide strategic insights, and ensure the final selection of companies "
#         "aligns with the defined financial metrics and investment goals. "
#         "Monitor the performance of the selected companies over the years 2022 and 2023, making strategic adjustments as needed."
#     ),
#     expected_output=(
#         "1. Final Outputs: \n"
#         "- Top five industries identified by the Industry Analyzer Agent \n"
#         "- 'Good' values defined by the Metrics Evaluation Agent for each industry \n"
#         "- Top ten companies selected by the Company Selection Agent for each industry \n"
#         "- Top three companies per industry chosen by the Top Picks Analysis Agent \n"
#         "2. Markdown Report: \n"
#         "- Comprehensive markdown report including the final outputs of all agents and a summary. \n"
#         "3. Summary: \n"
#         "- Overview of the methodology used by each agent \n"
#         "- Key findings and insights from the analysis \n"
#         "- Performance summary of the selected companies for the years 2022 and 2023 \n"
#         "- Strategic recommendations for future investments"
#     ),
#     agent=strategic_investment_system_manager,
# )


industry_analyzer_agent = Agent(
    role="Industry Analyzer Agent",
    goal="Analyze historical data from 2017 to 2021 to identify the top five performing industries based on "
         "robust industry performance metrics.",
    backstory="This agent works within a prestigious financial research firm and has a deep understanding "
              "of market trends and industry performance indicators. "
              "It holds a background in Data Analysis and Financial Analysis, with proficiency in statistical"
              " software and programming languages like Python and R. "
              "The agent is adept at interpreting financial data to identify industry trends and rank industry performance.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool, CalculatorTools.calculate]
)

industry_analysis_task = Task(
    description=(
        "Analyze historical financial data from 2017 to 2021 to identify the top five performing industries. "
        "Evaluate each industry based on the following five key metrics: "
        "Market Share Growth, Industry Revenue Growth, Industry Profit Margins, Return on Invested Capital (ROIC), "
        "and Industry Employment Growth. "
        "Generate a ranked list of industries based on their performance over the specified period. "
        "Submit the list to the Strategic Investment System Manager for review and approval."
    ),
    expected_output=(
        "A detailed report identifying the top five industries from 2017 to 2021, including: \n"
        "1. The ranked list of industries based on performance metrics \n"
        "2. A summary of the financial metrics and statistical methods used for the analysis \n"
        "3. Insights and trends observed within the top-performing industries \n"
        "4. Detailed calculations and data for the five key metrics: Market Share Growth, "
        "Industry Revenue Growth, Industry Profit Margins, Return on Invested Capital (ROIC), and Industry Employment Growth"
    ),
    agent=industry_analyzer_agent,
)

metrics_evaluation_agent = Agent(
    role="Metrics Evaluation Agent",
    goal="Define 'good' values for six key financial metrics for companies within each of the top-performing "
         "industries identified by the Industry Analyzer Agent.",
    backstory="This agent holds an advanced degree in Finance or Economics and certifications like Chartered "
              "Financial Analyst (CFA). "
              "The agent has extensive experience in corporate financial evaluation and benchmarking. "
              "It applies advanced financial analysis techniques "
              "to set benchmarks for key financial metrics, ensuring that the selected companies meet "
              "high standards of financial health and performance.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool, CalculatorTools.calculate]
)

metrics_evaluation_task = Task(
    description=(
        "Establish specific 'good' values for six key financial metrics for companies within each industry "
        "identified by the Industry Analyzer Agent. "
        "The six key metrics are Revenue Growth, Profit Margins (Gross/Operating/Net), Return on Equity (ROE), "
        "Debt-to-Equity Ratio, "
        "Price-to-Earning Ratio (P/E), and Free Cash Flow (FCF). Ensure these values align with industry "
        "standards and historical performance data. "
        "Provide the criteria to the Strategic Investment System Manager for confirmation and refinement."
    ),
    expected_output=(
        "A comprehensive report defining the 'good' values for the six key financial metrics for companies "
        "within each of the top industries, including: \n"
        "1. Defined 'good' values for Revenue Growth, Profit Margins (Gross/Operating/Net), Return on "
        "Equity (ROE), Debt-to-Equity Ratio, Price-to-Earning Ratio (P/E), and Free Cash Flow (FCF) for each industry \n"
        "2. Justification for each value based on historical data and industry standards \n"
        "3. Detailed explanation of the methodology used to determine these values"
    ),
    agent=metrics_evaluation_agent,
)

company_selection_agent = Agent(
    role="Company Selection Agent",
    goal="Filter and select the top ten companies in each of the best industries based on the defined metrics "
         "provided by the Metrics Evaluation Agent.",
    backstory="This agent has a background in Investment Banking or Corporate Finance with strong analytical "
              "skills and experience in financial databases and tools. "
              "The agent is knowledgeable in equity research and quantitative methods, enabling it to "
              "effectively filter and select top-performing companies based on rigorous criteria.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool, CalculatorTools.calculate]
)

company_selection_task = Task(
    description=(
        "Apply the criteria defined by the Metrics Evaluation Agent to filter and select the top ten companies "
        "from each of the best industries identified by the Industry Analyzer Agent. "
        "Ensure that the selected companies meet the 'good' values for the six key financial metrics: Revenue Growth, "
        "Profit Margins (Gross/Operating/Net), Return on Equity (ROE), Debt-to-Equity Ratio, "
        "Price-to-Earning Ratio (P/E), and Free Cash Flow (FCF). Provide the list of selected companies to the "
        "Strategic Investment System Manager for a strategic review."
    ),
    expected_output=(
        "A detailed list of the top ten companies for each of the top industries based on the defined metrics, "
        "including: \n"
        "1. The top ten companies selected for each industry \n"
        "2. A summary of the evaluation process and criteria applied \n"
        "3. Justification for the selection of each company based on the six key financial metrics"
    ),
    agent=company_selection_agent,
)

top_picks_analysis_agent = Agent(
    role="Top Picks Analysis Agent",
    goal="Further analyze the top ten companies in each of the best industries to select the top three companies, "
         "ensuring the best candidates for long-term investment.",
    backstory="This agent has extensive experience in Portfolio Management or Equity Analysis, with expertise "
              "in using backtesting software and investment strategies. "
              "The agent holds certifications in portfolio management such as Certified Financial Planner (CFP)"
              " or Chartered Investment Manager (CIM). It conducts in-depth analysis to identify the most "
              "promising companies for investment based on financial health and growth potential.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool, CalculatorTools.calculate]
)

top_picks_analysis_task = Task(
    description=(
        "Conduct a thorough financial and growth potential analysis of the top ten companies in each industry "
        "selected by the Company Selection Agent. "
        "Identify the top three companies in each industry based on a deeper financial analysis and potential "
        "for growth. Ensure that these companies are the best candidates for long-term investment. "
        "Provide the final list of top companies to the Strategic Investment System Manager for final "
        "validation and oversight."
    ),
    expected_output=(
        "A detailed list of the top three companies for each of the top industries based on a deeper "
        "analysis, including: \n"
        "1. The top three companies selected for each industry \n"
        "2. A summary of the in-depth evaluation process and criteria applied \n"
        "3. Justification for the final selection of each company based on financial health and growth potential"
    ),
    agent=top_picks_analysis_agent,
)

manager = Agent(
    role="Manager",
    goal="Output the best 3 US companies (in S&P 500) in each of top 5 industries in the period 2017-2021. "
         "The last output must be in markdown format. "
         "Oversee the multi-agent system designed "
         "to backtest long-term investment strategies. Ensure each agent operates efficiently "
         "and collaboratively to identify and analyze top-performing industries and companies "
         "based on defined financial metrics. ",
    backstory="This agent holds a Master’s degree in Finance and an MBA, along with CFA and CIMA certifications. "
              "With a proven track record in investment management and strategic planning, the agent has extensive experience in overseeing data-driven projects and teams within the financial sector. "
              "The agent is responsible for strategic decision-making, validating outputs, and ensuring the system's alignment with broader investment goals.",
    verbose=True,
    llm=ChatOpenAI(model=MANAGER_MODEL, temperature=0.05),
    allow_delegation=True,
)
crew = Crew(
    agents=[industry_analyzer_agent, metrics_evaluation_agent,
            company_selection_agent, top_picks_analysis_agent],
    tasks=[industry_analysis_task, metrics_evaluation_task,
           company_selection_task, top_picks_analysis_task],
    manager_agent=manager,
    process=Process.hierarchical,
    verbose=True,
)

result = crew.kickoff()

print("######################")
print(result)
