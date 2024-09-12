import os

from crewai import Agent, Task, Crew, Process
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_openai import ChatOpenAI

MANAGER_MODEL = "gpt-4-turbo"
AGENT_MODEL = "gpt-4o-mini"
load_dotenv()
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
search = TavilySearchAPIWrapper()
tavily_search_tool = TavilySearchResults(api_wrapper=search, max_results=5)

os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = AGENT_MODEL
political_agent = Agent(
    role="Political Agent",
    goal="Gather and analyze political developments related to the Communist Party of the Philippines (CPP), "
         "including government actions, public statements, and peace negotiations using search and scrape tools.",
    backstory="The Political Agent is a specialized researcher with expertise in political science, particularly in "
              "insurgent and revolutionary movements. This agent relies solely on search engines and web scraping "
              "techniques to gather relevant data from political news outlets, government websites, and public platforms.",
    verbose=True,
    allow_delegation=False,
    tools=[tavily_search_tool]
)

political_task = Task(
    description=(
        "Use search and scraping tools to gather the latest news and political reports about CPP-related activities. "
        "Track government actions, public statements, and peace talks, particularly those involving the CPP. "
        "Collect information on political alliances and responses to CPP activities. "
        "Some main points: \n"
        "\t- The role of the Chinese Government in relation to the CPP and NPA, including any official or unofficial support/funding. \n"
        "\t- Identifying the main areas where the CPP/NPA are operating. \n"
        "\t- How ordinary citizens are impacted by and perceive the activities of the CPP/NPA. \n"
        "\t- Member counts of the CPP. \n"
        "Ensure all information is backed by references from trusted sources and websites."
    ),
    expected_output=(
        "A detailed political summary report of the CPP’s current political state, including: \n"
        "1. A summary of government actions, public statements, and negotiations involving the CPP \n"
        "2. Details on political alliances and movements affecting the CPP \n"
        "3. All references and sources, including URLs from trusted and verified websites"
        "4. The number for any data should be provided if possible\n"
    ),
    agent=political_agent,
)

military_agent = Agent(
    role="Military Agent",
    goal="Gather and analyze military developments related to the Communist Party of the Philippines (CPP), "
         "including armed conflicts, ceasefires, and military strategies using search and scrape tools. "
         "Ensure all data is verified by the Evaluator Agent before proceeding.",
    backstory="The Military Agent is a military analyst with expertise in guerrilla warfare, counterinsurgency, and "
              "military strategy. The agent uses search engines and web scraping tools to collect relevant data from "
              "military news sources, government defense websites, and conflict monitoring platforms. The agent identifies "
              "significant patterns in CPP military activities and organizational structures. All data is validated "
              "by the Evaluator Agent before generating summaries.",
    verbose=True,
    allow_delegation=True,
    tools=[tavily_search_tool]
)

military_task = Task(
    description=(
        "Use search and scraping tools to gather the latest news and reports on CPP military actions, armed conflicts, "
        "and government military responses. Track the status of ceasefires, peace agreements, and ongoing military operations "
        "involving the CPP. Analyze military strategies, organizational structures, and the impact on local populations. "
        "Some main points: \n"
        "\t- The role of the Chinese Government in relation to the CPP and NPA, including any official or unofficial support/funding. \n"
        "\t- Identifying the main areas where the CPP/NPA are operating. \n"
        "\t- How ordinary citizens are impacted by and perceive the activities of the CPP/NPA. \n"
        "\t- Member counts of the NPA. \n"
        "Delegate all collected data to the Evaluator Agent for validation before preparing summaries."
    ),
    expected_output=(
        "A detailed military summary report of the CPP’s current military state, including: \n"
        "1. Ongoing conflicts, ceasefires, and peace agreements involving the CPP \n"
        "2. Government military actions and responses against the CPP \n"
        "3. Analysis of CPP military strategies and key military figures \n"
        "4. All references and sources, including URLs from trusted military and government websites, validated by the Evaluator Agent\n"
        "5. The number for any data should be provided if possible\n"
    ),
    agent=military_agent,
)

socio_economic_agent = Agent(
    role="Socio-Economic Agent",
    goal="Gather and analyze socio-economic data related to the Communist Party of the Philippines (CPP), "
         "including the impact of CPP activities on rural and urban populations, economic conditions, and societal structures. "
         "Ensure all data is verified by the Evaluator Agent before proceeding.",
    backstory="The Socio-Economic Agent has expertise in economics, sociology, and development studies. "
              "The agent uses search engines and web scraping tools to collect data from government reports, economic databases, "
              "and news sources that track economic and social conditions. The agent identifies patterns of economic disruption "
              "and social change caused by the CPP insurgency and government responses, particularly in rural areas. All data "
              "is validated by the Evaluator Agent before generating summaries.",
    verbose=True,
    allow_delegation=True,
    tools=[tavily_search_tool]
)

socio_economic_task = Task(
    description=(
        "Use search and scraping tools to gather socio-economic data on the CPP’s impact, focusing on how their activities "
        "affect communities in terms of displacement, employment, and access to resources. Track economic trends in rural and urban "
        "areas affected by the CPP and analyze social structures, such as migration and healthcare access. Collect government and NGO "
        "reports on socio-economic conditions in conflict-affected regions. "
        "Some main points: \n"
        "\t- The role of the Chinese Government in relation to the CPP and NPA, including any official or unofficial support/funding. \n"
        "\t- Identifying the main areas where the CPP/NPA are operating. \n"
        "\t- How ordinary citizens are impacted by and perceive the activities of the CPP/NPA. \n"
        "\t- Member counts of the NPA. \n"
        "Delegate all collected data to the Evaluator Agent for validation."
    ),
    expected_output=(
        "A detailed socio-economic summary report on the CPP’s current impact, including: \n"
        "1. Economic effects in rural and urban areas, focusing on employment, trade, and resource access \n"
        "2. Social impact, including migration, education, and healthcare access in conflict-affected areas \n"
        "3. All references and sources, including URLs from trusted government, NGO, and economic reports, validated by the Evaluator Agent\n"
        "4. The number for any data should be provided if possible\n"
    ),
    agent=socio_economic_agent,
)

global_perspective_agent = Agent(
    role="Global Perspective Agent",
    goal="Gather and analyze global and international data related to the Communist Party of the Philippines (CPP), "
         "including international diplomatic responses, foreign policy decisions, and global human rights reports. "
         "Ensure all data is verified by the Evaluator Agent before proceeding.",
    backstory="The Global Perspective Agent is a specialist in international relations with a focus on geopolitics and conflict studies. "
              "It uses search engines and web scraping tools to collect data from international news outlets, global think tanks, and "
              "governmental organizations. The agent tracks international diplomatic responses, foreign military involvement, and human rights "
              "reports regarding the CPP’s activities. All collected data is validated by the Evaluator Agent before generating summaries.",
    verbose=True,
    allow_delegation=True,
    tools=[tavily_search_tool]
)

global_perspective_task = Task(
    description=(
        "Use search and scraping tools to gather global data on the CPP’s international reputation, focusing on foreign government responses, "
        "international relations, and diplomatic efforts. Analyze foreign policy decisions and global human rights reports concerning the CPP. "
        "Track any foreign military involvement or international sanctions related to the CPP insurgency. "
        "Some main points: \n"
        "\t- The role of the Chinese Government in relation to the CPP and NPA, including any official or unofficial support/funding. \n"
        "\t- Identifying the main areas where the CPP/NPA are operating. \n"
        "\t- How ordinary citizens are impacted by and perceive the activities of the CPP/NPA. \n"        
        "Delegate all collected data to the Evaluator Agent for validation before preparing summaries."
    ),
    expected_output=(
        "A detailed global perspective summary report on the CPP’s current international impact, including: \n"
        "1. Foreign government responses, international policies, and sanctions concerning the CPP \n"
        "2. Reports from global human rights organizations on the CPP’s activities \n"
        "3. All references and sources, including URLs from trusted global news outlets, governmental organizations, and think tanks, "
        "validated by the Evaluator Agent\n"
        "4. The number for any data should be provided if possible\n"
    ),
    agent=global_perspective_agent,
)

evaluator_agent = Agent(
    role="Evaluator Agent",
    goal="Verify and validate all data collected by the specialized agents (Political, Military, Socio-Economic, Global) "
         "before it is compiled into summaries. Additionally, enhance the summaries by searching and scraping more numerical data to create a refined text. "
         "Ensure the accuracy, credibility, and completeness of all information before use.",
    backstory="The Evaluator Agent is an expert in data validation and research methodologies. It ensures the integrity of information "
              "gathered by other agents by using analytical skills to cross-check data against reliable, trusted sources. The agent also seeks additional numerical data "
              "to enrich the final summaries. The Evaluator Agent rejects any unverified or questionable data to maintain the credibility of the overall report.",
    verbose=True,
    allow_delegation=False,
    tools=[tavily_search_tool, search_tool, scrape_tool]  # Placeholder tools
)

evaluator_task = Task(
    description=(
        "Review and validate all data collected by the Political, Military, Socio-Economic, and Global Perspective Agents. "
        "Ensure that each piece of information is accurate, relevant, and backed by credible references. Additionally, search for more numerical data "
        "through scraping or other methods and integrate this information into the summaries to create an enhanced version. "
        "Cross-check the data with external trusted sources and approve or reject data submissions based on validation criteria.\n"
        "IMPORTANT: Avoid processing PDF files, only validate information sourced from web pages."
    ),
    expected_output=(
        "1. A validated set of summaries from all agents (Political, Military, Socio-Economic, Global), enhanced with additional numerical data. \n"
        "2. References to trusted sources that support and validate the enhanced information. \n"
        "3. A refined text containing both validated summaries and new numerical insights, ready for the Narrative Generator to create the final report. \n"
        "4. Numerical data should be provided wherever applicable and sought out if missing."
    ),
    agent=evaluator_agent,
)

narrative_generator_agent = Agent(
    role="Narrative Generator Agent",
    goal="Collecting the summaries from the specialized agents into a cohesive and comprehensive report ~ 1000 words. "
         "Present a clear and insightful narrative on the current state of the Communist Party of the Philippines (CPP) "
         "based solely on the summaries provided by the Political, Military, Socio-Economic, and Global agents (after edited/augmented by Evaluator).",
    backstory="The Narrative Generator Agent is skilled in report writing, with experience in journalism and academic writing. "
              "It specializes in transforming multi-faceted data into a clear and engaging narrative. "
              "The agent focuses on synthesizing information from the four specialized agents, presenting a full picture of the CPP’s "
              "current status without the need for external tools or additional validation.",
    verbose=True,
    allow_delegation=False,  # No delegation needed, as it works directly with summaries
    tools=[]  # No external tools required
)

narrative_task = Task(
    description=(
        "Collect the summaries & write article of ~ 1000 words provided by the Political, Military, Socio-Economic, and Global agents (after edited/augmented by Evaluator Agent) into a single comprehensive report. "
        "The report must cover the historical context, military actions, socio-economic impacts, and global views regarding the CPP. "
        "Ensure that the report is accessible, well-structured, and covers all perspectives provided by the specialized agents."
    ),
    expected_output=(
        "A detailed narrative report on the current state of the CPP, including: \n"
        "1. A synthesis of political, military, socio-economic, and global perspectives \n"
        "2. Insights into the CPP’s history, ideology, and recent activities \n"
        "3. A final, coherent, and engaging report suitable for public or academic consumption\n"
        "4. The number for important data should be provided \n"
    ),
    agent=narrative_generator_agent,
)

manager_agent = Agent(
    role="Manager Agent",
    goal="Coordinate the tasks of the Political, Military, Socio-Economic, and Global Perspective agents. "
         "Serve as a senior academic researcher, guiding the questioning and research directions based on academic rigor. "
         "Ensure that each agent effectively contributes its part and that the multi-agent system operates efficiently. "
         "Focus on specific research points, including: \n"
         "- The role of the Chinese Government in relation to the CPP and NPA, including any official or unofficial support/funding. \n"
         "- Identifying the main areas where the CPP/NPA are operating. \n"
         "- How ordinary citizens are impacted by and perceive the activities of the CPP/NPA. \n"
         "Additionally, gather specific numbers, such as: \n"
         "- Member counts of the CPP. \n"
         "- Member counts of the NPA.",
    backstory="The Manager Agent holds advanced academic credentials and has a deep understanding of the Philippines' complex political, "
              "socio-economic, and military landscape, particularly with respect to insurgent groups like the Communist Party of the Philippines (CPP). "
              "With expertise in Marxist-Leninist-Maoist movements, this agent tracks the long history of the CPP, which was founded in 1968 and aims to overthrow the government "
              "and establish a socialist state. It understands the CPP’s reliance on a combination of military struggle and mass mobilization, particularly in rural areas, "
              "and is well-versed in government responses, peace talks, and international perspectives on the insurgency.",
    verbose=True,
    llm=ChatOpenAI(model=MANAGER_MODEL, temperature=0),  # Placeholder for the model used
    allow_delegation=True,
)

crew = Crew(
    agents=[political_agent, military_agent, socio_economic_agent, global_perspective_agent, evaluator_agent,
            narrative_generator_agent],
    tasks=[political_task, military_task, socio_economic_task, global_perspective_task, evaluator_task, narrative_task],

    manager_agent=manager_agent,
    process=Process.hierarchical,
    verbose=True,
)

result = crew.kickoff()
print("######################")
print(result)
