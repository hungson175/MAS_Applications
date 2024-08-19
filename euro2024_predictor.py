import os
import warnings
from dotenv import load_dotenv, find_dotenv

warnings.filterwarnings('ignore')
from crewai import Agent, Task
from langchain_openai import ChatOpenAI

MANAGER_MODEL = "gpt-3.5-turbo"
AGENT_MODEL = "gpt-3.5-turbo"
# IP_ADDRESS = "http://192.168.0.223"

# IP_ADDRESS = "http://localhost"
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

fd_aa = Agent(
    role="Data Analyst",
    goal="Monitor and analyze football data in real-time to identify trends and predict match outcomes.",
    backstory="Equipped with advanced analytics capabilities, this agent uses statistical modeling and "
              "machine learning to analyze data from football matches. It specializes in extracting actionable "
              "insights from complex datasets, focusing on team formations, player performances, "
              "and historical trends to predict outcomes of football matches.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool]
)

data_analysis_task = Task(
    description=(
        "Monitor and analyze football data for "
        "the final match ({match_selection}) of UEFA Euro 2024. Pay attention VERY closely to the latest news, "
        "especially: recent matches performance, injuries & yellow/red cards. "
        "Utilize advanced statistical modeling and machine learning to "
        "identify performance trends, assess team strategies, and predict the outcome of the match. "
        "Focus on player impact forecasts, specific match predictions, and tactical recommendations."
    ),
    expected_output=(
        "Comprehensive insights including detailed predictions of the selected match outcome, "
        "key player performance analyses, and strategic adjustments likely to be made by teams. "
        "Provide alerts on significant trends and potential game-changing factors such as player injuries, "
        "recent matches performance, and tactical shifts."
    ),
    agent=fd_aa,
)

player_analyst_agent = Agent(
    role="Player Analyst",
    goal="Analyze key players’ performances based on data and predictions provided by the "
         " Data Analyst Agent (DAA), supplemented with targeted web searches.",
    backstory="Embodying the analytical acumen and in-depth understanding of football "
              "dynamics akin to Gary Neville, this agent brings expert-level analysis into player performance from "
              "a player perspective. Primarily relying on data from the DAA, "
              "it mirrors Neville’s approach to enhancing "
              "insight with tactical knowledge and up-to-date information on "
              "player conditions and strategic deployments.",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
)

player_analysis_task = Task(
    description=(
        "Utilizing the analytical style of Gary Neville, focus on key players identified by the "
        "Data Analyst Agent (DAA) for the final match ({match_selection}) of UEFA Euro 2024. "
        "Analyze these players' current form, tactical roles, and potential impact, "
        "using foundational data from DAA enhanced with selective web searches for the latest updates."
    ),
    expected_output=(
        "Detailed analysis reports on key players who are expected to be crucial in determining the match outcome. "
        "These reports will discuss the players’ strengths, weaknesses, tactical fit, and expected contributions, "
        "reflecting Gary Neville’s insightful and strategic commentary style. Each report will also include "
        "predictions on these players' influence on the game, "
        "with tactical insights and potential game-changing moments."
    ),
    agent=player_analyst_agent,
)

coach_analyst_agent = Agent(
    role="Coach Analyst",
    goal="Predict and formulate the most effective strategies for the upcoming match,"
         " considering the recent matches provided by Data Analyst and "
         "player insights provided by the Player Perspective Agent.",
    backstory="Modeled after Sir Arsène Wenger, this agent combines a deep understanding"
              " of football dynamics with a meticulous approach to match preparation. "
              "Known for his foresight and tactical innovation, the agent uses data-driven "
              "analysis to anticipate game scenarios and devise strategies that optimize both "
              "teams performance. Reflecting Wenger’s articulate and thoughtful analysis style, "
              "it delivers strategic predictions essential for overcoming opponents and securing "
              "a favorable position in the tournament.",
    verbose=True,
    allow_delegation=True
)

coach_strategy_task = Task(
    description=(
        "Integrate detailed recent matches data from the Data Analyst and critical "
        "player insights from the Player Perspective to predict and "
        "plan the best tactical approaches for the "
        "final match ({match_selection}) of UEFA Euro 2024. Assess the potential impacts of various strategic "
        "decisions on the match outcome. Request additional contextual data from FDAA "
        "or specific player details from PA as necessary to refine predictions."
    ),
    expected_output=(
        "A predictive report that outlines tactical strategies likely to succeed"
        " based on current data analysis. The report will focus on optimizing team formation,"
        " player roles, and in-game tactics to enhance the team's chances of winning or "
        "achieving the necessary result to advance. Each strategy will be evaluated for "
        "its potential to influence key aspects of the match, ensuring readiness for various match scenarios."
    ),
    agent=coach_analyst_agent,
)

head_of_sportsbook = Agent(
    role="Head of Sportsbook",
    goal="Use integrated data from Data Analyst Agent, Player Perspective Agent, and Coach Perspective Agent "
         "to set final betting odds and provide detailed match analyses for any given match.",
    backstory="As the final decision-maker in sports betting operations, this agent synthesizes "
              "expert knowledge in football analytics, betting trends, and risk management to predict"
              " match outcomes and set competitive odds. Specialized in Euro 2024, the agent adapts"
              " its predictions and analyses to any specific match, ensuring that all betting odds "
              "reflect a deep understanding of team dynamics, player performances, and tactical setups.",
    verbose=True,
    allow_delegation=True
)

sportsbook_task = Task(
    description=(
        "Integrate and analyze predictions and data from the Data Analyst Agent, Player Perspective Agent, "
        "and Coach Perspective Agent for the final match ({match_selection}) of UEFA Euro 2024. "
        "Calculate the probabilities for each team to win, draw, or lose and provide a "
        "comprehensive analysis for each possible outcome."
    ),
    expected_output=(
        "1. Match Outcome Probabilities: Detailed percentages indicating the likelihood of "
        "each outcome (win for either team) for the final match ({match_selection}) of UEFA Euro 2024. "
        "2. Match Analysis: Provide an in-depth analysis for each scenario (win, lose)"
        " for the final match ({match_selection}), explaining the key factors "
        "and statistics that influence these outcomes, including tactical insights, "
        "player impacts, and strategic decisions."
    ),
    agent=head_of_sportsbook,
)

from crewai import Crew, Process
from langchain_openai import ChatOpenAI
football_analysis_crew = Crew(
    agents=[fd_aa, player_analyst_agent, coach_analyst_agent, head_of_sportsbook],
    tasks=[data_analysis_task, player_analysis_task, coach_strategy_task, sportsbook_task],
    manager_llm=ChatOpenAI(model=MANAGER_MODEL, temperature=0.2),
    process=Process.hierarchical,
    verbose=True
)

football_match_inputs = {
    'match_selection': 'Spain vs. England',
    'news_impact_consideration': True,
}

result = football_analysis_crew.kickoff(inputs=football_match_inputs)