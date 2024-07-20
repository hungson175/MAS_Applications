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
copy_writer = Agent(
    role="Copywriter",
    goal="Dựa vào các điểm bán hàng chính về một dự án bất động sản tại Việt Nam, bạn"
         " phải viết một bài viết tiếp thị cho dự án đó (Tất cả bằng tiếng Việt)",
    backstory="""Hãy đóng vai một người viết bài quảng cáo chuyên nghiệp, là người:\n
    - Người Việt bản địa nhưng rất giỏi tiếng Anh \n
    - Biết sử dụng nguyên tắc EEAT để viết bài \n
    - Có kiến thức sâu rộng về bất động sản tại Việt Nam\n""",
    verbose=True,
    memomry=True,
    tools=[search_tool, scrape_tool],
);

copy_writer_task = Task(
    description="Viết bài quảng cáo cho dự án bất động sản {project_name} tại Việt Nam,"
                "theo các bước sau:\n"
                "Bước 1. Tìm hiểu về dự án bất động sản {project_name} tại Việt Nam qua link {site_url} \n"
                "Bước 2. Viết bản báo cáo đầy đủ về thông tin dự án {project_name} dựa trên link ở trên \n"
                "Bước 3. Dựa trên bản báo cáo đầy đủ, viết bản tóm tắt về dự án dưới dạng gạch đầu dòng \n"
                "Bước 4. Xác định đối tượng bài viết là nhà đầu tư nhỏ lẻ, họ quan tâm nhất tới mục đích sinh lời"
                "nhưng vẫn phải an toàn, vì thế đây là 5 tiêu chí chính để đánh giá tiềm năng của một dự án BĐS:\n"
                "  4.1. Vị trí dự án và phát triển hạ tầng: Vị trí của bất động sản luôn là yếu tố quan trọng"
                " nhất. Vị trí đắc địa gần trung tâm thành phố, các khu vực có tiềm năng phát triển du lịch, gần"
                " các tiện ích như trường học, bệnh viện, trung tâm thương mại sẽ giúp tăng giá trị bất động sản"
                " theo thời gian. Quy hoạch rõ ràng và phát triển hạ tầng đồng bộ là yếu tố quan trọng. Những khu"
                " vực được đầu tư phát triển hạ tầng giao thông, tiện ích công cộng sẽ có giá trị bất động sản"
                " tăng cao hơn \n"
                "  4.2. Tiềm năng giá và thanh khoản: Nhà đầu tư cần xem xét tiềm năng tăng giá của khu vực đó "
                "trong tương lai. Các dự án có quy hoạch rõ ràng, hạ tầng giao thông phát triển, và chính sách"
                " phát triển của chính quyền địa phương thường có tiềm năng tăng giá cao hơn. Khả năng mua bán lại"
                " bất động sản một cách dễ dàng và nhanh chóng là yếu tố quan trọng. Những dự án có tính thanh"
                " khoản cao sẽ giúp nhà đầu tư dễ dàng thoát hàng khi cần và tối ưu hóa lợi nhuận. Các chính sách"
                " hỗ trợ tài chính từ chủ đầu tư, như hỗ trợ vay vốn, chính sách thanh toán linh hoạt, và các"
                " chương trình ưu đãi đặc biệt sẽ giảm áp lực tài chính cho nhà đầu tư và giúp họ dễ dàng tiếp cận"
                " bất động sản hơn\n"
                "  4.3. Uy tín chủ đầu tư và tính pháp lý:Nhà đầu tư cần chọn các dự án từ những chủ đầu tư uy"
                " tín, có kinh nghiệm và năng lực triển khai dự án đúng tiến độ và chất lượng cam kết. Điều này"
                " giúp giảm rủi ro và đảm bảo an toàn cho khoản đầu tư của họ. Tính pháp lý của dự án là yếu tố"
                " không thể bỏ qua. Nhà đầu tư cần đảm bảo dự án có đầy đủ giấy tờ pháp lý, sổ đỏ, giấy phép xây"
                " dựng để tránh rủi ro tranh chấp và bảo vệ quyền lợi của mình \n "
                " 4.5. Tiện ích và dịch vụ: Những dự án có nhiều tiện ích và dịch vụ đẳng cấp sẽ thu hút nhiều "
                "người mua và thuê, từ đó tăng giá trị bất động sản và khả năng khai thác lợi nhuận từ việc cho"
                " thuê \n"
                "Bước 5: Viết bài bằng cách phân tích đầu tư khéo léo, chứ không đi thẳng vào giới thiệụ dự án dàn ý như sau:\n"
                "  5.1. Phân tích xu hướng thị trường BĐS trong những năm gần đây: Bài viết thường bắt đầu bằng việc phân tích tình hình hiện tại của thị trường bất động sản, bao gồm các yếu tố như cung cầu, biến động giá cả, chính sách của nhà nước, và các xu hướng đầu tư mới nhất. \n"
                "  5.2. Lợi ích của việc đầu tư vào bất động sản hiện hữu: Bài viết nhấn mạnh vào lợi ích của việc đầu tư vào các bất động sản hiện hữu so với các dự án mới, chẳng hạn như tính ổn định, ít rủi ro pháp lý, và khả năng tạo ra thu nhập thụ động từ việc cho thuê.\n"
                "  5.3. Phân tích các yếu tố quyết định thành công: Bài viết sẽ phân tích chi tiết các yếu tố quan"
                " trọng mà nhà đầu tư cần xem xét, chẳng hạn như vị trí, tiềm năng tăng giá, tính thanh khoản, và"
                " các tiện ích xung quanh. Các yếu tố này giúp nhà đầu tư đánh giá được mức độ hấp dẫn của một dự"
                " án bất động sản.\n"
                "  5.4.Giới thiệu khéo léo về dự án cụ thể: Bài viết sẽ lồng ghép thông tin về các dự án bất động"
                " sản {project_name} một cách tự nhiên và khéo léo. Thay vì quảng cáo trực tiếp, thông tin về dự"
                " án được giới thiệu như là một ví dụ điển hình cho các nguyên tắc đầu tư mà bài viết đã nêu ra."
                " Điều này giúp tăng độ tin cậy và thuyết phục người đọc hơn.\n"
                "  5.5. Kết luận - Lời khuyên và chiến lược đầu tư: Cuối bài viết, thường có phần đưa ra các lời"
                " khuyên và chiến lược đầu tư cụ thể cho người đọc. Các chiến lược này dựa trên phân tích thị"
                " trường và những yếu tố đã đề cập trước đó, giúp nhà đầu tư có thể áp dụng vào thực tế\n"
    ,
    expected_output="Bài viết bằng tiếng Việt, dưới dạng markdown được viết theo các bước đã mô tả",
    agent=copy_writer,
    verbose=True,
);

crew = Crew(
    agents=[copy_writer],
    tasks=[copy_writer_task],
    manager_llm=ChatOpenAI(model=MANAGER_MODEL, temperature=0.2),
    process=Process.hierarchical,
    verbose=True
);

inputs = {
    "project_name": "Novawold Phan Thiết",
    "site_url": "https://www.novaland.com.vn/novaworld-phan-thiet-hoi-tu-du-dieu-kien-tro-thanh-khu-do-thi-kinh-te-du-lich-quoc-te"
}
crew.kickoff(inputs=inputs)
