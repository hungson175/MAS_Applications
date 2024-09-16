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


def get_report_in_vietnamese(query: str, report_type: str) -> dict:
    report = asyncio.run(generate_report(query, report_type))
    llm = ChatOpenAI(model_name="gpt-4o")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are very skillful translator. 
         Your goal is to translate the following report into Vietnamese - make sure to keep the meaning,tone, and references of the report intact."""),
        # this will be auto generate from query
        ("human", "{report}")
    ])
    translator = prompt_template | llm
    result = translator.invoke(input={"report": report})
    return {
        "en": report,
        "vi": result.content
    }


def generate_file_name(query: str):
    system_prompt = """Given the query, your task is to generate a file name for the report using 3-5 words.
    Expected output: The output file name WITHOUT extension.

    Examples:
        query: "The Impact of Substances on Creativity and Innovation Throughout History"
        output: "impact_substances_creativity_innovation_history"

        query: "How does the brain process information?"
        output: "brain_process_information"

        query: "What is the impact of AI on the job market?"
        output: "impact_ai_job_market"
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.15)
    chain = prompt | llm
    return chain.invoke({"query": query})


if __name__ == "__main__":
    # query = "The Impact of AI on Software Outsourcing Companies: Strategies for Adaptation"
    # query = "Adapting to the Impact of Generative AI: How HR Managers Can Stay Relevant ?"
    query = "Adapting to the Impact of Generative AI: How Software Developers can stay relevant ?"
    file_prefix = generate_file_name(query)
    english_file_name = file_prefix.content + "_en.md"
    vietnamese_file_name = file_prefix.content + "_vi.md"

    report_type = "research_report"
    report = get_report_in_vietnamese(query, report_type)

    # save the report to files
    with open(english_file_name, "w") as f:
        f.write(report["en"])
    with open(vietnamese_file_name, "w") as f:
        f.write(report["vi"])

    print(f"Report saved to {english_file_name} and {vietnamese_file_name}")

