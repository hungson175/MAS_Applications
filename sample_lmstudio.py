import os
import openai
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

IP_ADDRESS = "http://192.168.0.223"

# IP_ADDRESS = "http://localhost"
# MODEL_NAME = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_NAME = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
load_dotenv(find_dotenv())

llm_lm_studio = ChatOpenAI(
    openai_api_base=IP_ADDRESS + ":1234/v1",
    openai_api_key="",
    model_name=MODEL_NAME,
    temperature=0.2,
)
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

researcher = Agent(
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science in',
    backstory="""You are a Senior Research Analyst at a leading tech think tank.
  Your expertise lies in identifying emerging trends and technologies in AI and
  data science. You have a knack for dissecting complex data and presenting
  actionable insights.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_lm_studio,
    tools=[search_tool, scrape_tool],
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""You are a renowned Tech Content Strategist, known for your insightful
  and engaging articles on technology and innovation. With a deep understanding of
  the tech industry, you transform complex concepts into compelling narratives.""",
    llm=llm_lm_studio,
    verbose=True,
    allow_delegation=False
)

task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. Your final answer MUST be a full analysis report""",
    expected_output="A detailed report outlining the latest advancements in AI in 2024,",
    agent=researcher
)

task2 = Task(
    description="""Using the insights from the researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. Your final answer MUST be the full blog post of at least 3 paragraphs.""",
    expected_output="An engaging blog post highlighting the most significant AI advancements in 2024,",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=True,
    process=Process.sequential
)

result = crew.kickoff()

print("######################")
print(result)
