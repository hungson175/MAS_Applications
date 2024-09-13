import asyncio

from backend.report_type import DetailedReport
from gpt_researcher.utils.enum import Tone
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

async def generate_report(query, report_type="research_report", report_source="web_search", tone=Tone.Formal,
                          websocket=None):
    detailed_report = DetailedReport(
        query=query,
        report_type=report_type,
        report_source=report_source,
        source_urls=[],  # You can provide initial source URLs if available
        # config_path="path/to/config.yaml",
        tone=tone,
        websocket=websocket,
        subtopics=[],  # You can provide predefined subtopics if desired
        headers={}  # Add any necessary HTTP headers
    )

    final_report = await detailed_report.run()
    return final_report


def get_report_in_vietnamese(query: str, report_type: str) -> str:
    report = asyncio.run(generate_report(query, report_type))
    llm = ChatOpenAI(model_name="gpt-4o")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are very skillful translator, who is working in tech industry. 
         Your goal is to translate the following report into Vietnamese"""),
        # this will be auto generate from query
        ("human", "{report}")
    ])
    translator = prompt_template | llm
    result = translator.invoke(input={"report": report})
    return result.content


if __name__ == "__main__":
    query = "Investment Analysis: Nafoods Group (NAF) Stock"
    report_type = "research_report"
    report = get_report_in_vietnamese(query, report_type)
    print(report)
