from dotenv import load_dotenv
import os

from gpt_researcher import GPTResearcher
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


async def get_report(query: str, report_type: str) -> str:
    researcher = GPTResearcher(query, report_type)
    research_result = await researcher.conduct_research()
    report = await  researcher.write_report()
    return report


def get_report_in_vietnamese(query: str, report_type: str) -> str:
    report = asyncio.run(get_report(query, report_type))
    llm = ChatOpenAI(model_name="gpt-4o")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         """You are very skillful translator, who is working in tech industry. Your goal is to translate the following report into Vietnamese"""),
        # this will be auto generate from query
        ("human", "{report}")
    ])
    translator = prompt_template | llm
    result = translator.invoke(input={"report": report})
    return result.content


if __name__ == "__main__":
    # query = "Best AI tools/IDE for developers"
    # report_type = "research_report"
    # res = get_report_in_vietnamese(query, report_type)
    # print(res)

    query = "Tôi có nên đầu tư cổ phiếu Nafoods Group (mã NAF)"
    # query = "Outline best AI tools/IDE for developers"
    # query = "List best AI tools/IDE for developers"
    report_type = "research_report"
    report = asyncio.run(get_report(query, report_type))
    print(report)
