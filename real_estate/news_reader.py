import os
import warnings

from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

MANAGER_MODEL = "gpt-4-turbo"
SMART_MODEL = "gpt-4-turbo"
AGENT_MODEL = "gpt-4-turbo"

load_dotenv()

# Access the API keys
os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = AGENT_MODEL

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

summary_agent = Agent(
    role="NVTH",
    goal="Từ một đường link website, tạo ra một bản tóm tắt ngắn gọn về nội dung của website đó bằng tiếng Việt.",
    backstory="Agent này nói tiếng Việt, rất am hiểu về thị trường bất động sản."
              "Chuyên đọc hiểu và tóm tắt nhanh chóng các thông tin quan trọng "
              "từ website của các dự án bất động sản (BĐS) .",
    verbose=True,
    llm=ChatOpenAI(model=SMART_MODEL, temperature=0.2),
    allow_delegation=False,
    tools=[scrape_tool]
)

summary_task = Task(
    description="Nhận một đường link website về dự án BĐS, tóm tắt nội dung chính của dự án đó."
                "Các bước thực hiện như sau:"
                "Bước 1: Sử dụng công cụ để tải nội dung website từ đường link: {website_url}"
                "Bước 2: Tách rõ ràng nội dung chính của website ra khỏi nội dung phụ như: "
                "đường link, quảng cáo, thông tin liên hệ ... "
                "Bước 3: Tóm tắt nội dung chính của dự án",
    expected_output="Tất cả bằng tiếng Việt: \n"
                    "1. Nội dung đầy đủ của dự án sau khi lọc bỏ thông tin không cần thiết"
                    "2. Tóm tắt nội dung chính của dự án dưới dạng các gạch đầu dòng, "
                    "mỗi gạch đâù dòng bao gồm thông tin, con số cụ thể về ý đó ",

    agent=summary_agent
)

real_estate_investor = Agent(
    role="NDT",
    goal="Đưa ra các tiêu chí về một dự án BĐS, đánh giá về tiềm năng sinh lời của dự án.",
    backstory="Nhà đầu tư lâu năm trong thị trường bất động sản Việt Nam, quan tâm nhất là khả năng sinh lời - "
              "biết rất rõ các tiêu chí để đánh giá một dự án BĐS có tiềm năng hay không, tránh được rủi ro, sinh lời "
              "bền vững ",
    verbose=True,
    allow_delegation=False,
)

real_estate_investor_task = Task(
    description="Đưa ra các tiêu chí của một dự án BĐS tốt",
    expected_output="Dùng tiếng Việt, đưa 5 tiêu chí của 1 dự án BĐS tốt tại Việt Nam",
    agent=real_estate_investor
);

copy_writer = Agent(
    role="BTV",
    goal="Tạo ra các bài viết chất lượng về bất động sản, viết bằng Tiếng Việt",
    backstory="Là người Việt Nam, một biên tập viên chuyên nghiệp, có kinh nghiệm viết về bất động sản, "
              "biết cách tạo ra các bài viết thu hút người đọc, tăng cường uy tín cho website. "
              "Biết cách áp dụng nguyên tắc EEAT."
              "Tốt nghiệp Đại học chuyên ngành Marketing và báo chí, làm biên tập viên báo/tạp chí về BĐS.",
    verbose=True,
    allow_delegation=True,
);

copy_writer_task = Task(
    description="Nhiệm vụ:"
                "Viết một bài viết tiếng Việt, nhằm marketing về dự án BĐS {real_estate_project} - "
                "với phong cách khéo léo bằng cách đưa lời khuyên chung về đầu tư BDS"
                ", chứ không trực tiếp quảng cáo dự án.\n"
                
                "Các bước để làm như sau:\n"
                "Bước 1: Nói chuyện với NDT để tìm hiểu về nhu cầu của khách hàng mục tiêu\n"
                
                "Bước 2: Giao cho NVTH tìm thông tin dự án từ website: {website_url}  \n"
                
                "Bước 3: Học phong cách viết từ một bài báo khác (phong cách, chứ không phải dự án đó) bằng cách "
                "giao cho NVTH lấy nội dung chính của bài viết cần học từ {writing_style_url} \n"
                
                "Bước 4: Đưa ra rõ ràng phong cách viết của bài báo ở bước 3 theo dạng gạch đầu dòng, "
                "lưu ý, chỉ cần học phong cách viết chứ không phải nội dung, không cần quan tâm nội dung."
                "Lưu ý rằng đây là phong cách đưa lời khuyên đầu từ tư góc độ có lợi cho nhà đầu tư cá nhân, "
                "rồi mới cài nội dung dự án cần quảng cáo vào ngắn gọn \n "
                
                "Bước 5: Tạo ra bài viết về dự án {real_estate_project} "
                "dựa trên cách thông tin đã thu thâp ở các bước trên",

    expected_output="Bài viết phải viết bằng Tiếng Việt, định dạng markdown",
    agent=copy_writer
);

crew = Crew(
    agents=[summary_agent, real_estate_investor, copy_writer],
    tasks=[copy_writer_task],
    manager_llm=ChatOpenAI(model=MANAGER_MODEL, temperature=0.2),
    process=Process.hierarchical,
    verbose=True
);
inputs = {
    'real_estate_project': 'NovaWorld Phan Thiết',
    'website_url': "https://www.novaland.com.vn/novaworld-phan-thiet-hoi-tu-du-dieu-kien-tro-thanh-khu-do-thi-kinh-te-du-lich-quoc-te",
    'writing_style_url': "https://batdongsan.com.vn/tin-tuc/gia-tri-dau-tu-cua-bat-dong-san-hien-huu-pr-806454"
}

results = crew.kickoff(inputs=inputs)

print("=========Results======")
print(results)
