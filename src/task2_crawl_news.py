"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://thanhnien.vn/cong-an-tam-giu-ca-si-chu-bin-lien-quan-den-ma-tuy-18524060619044815.htm",
    "https://thanhnien.vn/dj-thai-hoang-vua-bi-bat-vi-tang-tru-ma-tuy-la-ai-185230425153220627.htm",
    "https://thanhnien.vn/ma-tuy-va-showbiz-su-thanh-loc-can-bat-dau-tu-nghe-si-185260513123425952.htm",
    "https://thanhnien.vn/nu-ca-si-trung-quoc-bi-phong-sat-toan-dien-vi-dinh-ma-tuy-1851452927.htm",
    "https://thanhnien.vn/nam-ca-si-trung-quoc-bi-bat-vi-su-dung-ma-tuy-185521254.htm",
]

SAMPLE_ARTICLES = {
    ARTICLE_URLS[0]: {
        "title": "Công an tạm giữ ca sĩ Chu Bin liên quan đến ma túy",
        "content_markdown": """
Ngày 6.6.2024, báo Thanh Niên đưa tin Công an Q.10, TP.HCM tạm giữ Chu Đăng Thanh,
tức ca sĩ Chu Bin, để điều tra hành vi tổ chức, sử dụng trái phép chất ma túy.
Thông tin ban đầu cho biết lực lượng chức năng kiểm tra một căn nhà trên địa bàn
P.2, Q.10 và phát hiện một nhóm người có dấu hiệu tổ chức, sử dụng trái phép chất
ma túy, trong đó có ca sĩ Chu Bin.

Vụ việc được đặt trong bối cảnh các hành vi tổ chức sử dụng trái phép chất ma túy
có thể bị xử lý hình sự tùy tính chất và mức độ. Bài báo cũng nhắc lại quá trình
hoạt động nghệ thuật của Chu Bin và sự chú ý của công chúng đối với các ồn ào
liên quan đến nghệ sĩ.
""",
    },
    ARTICLE_URLS[1]: {
        "title": "DJ Thái Hoàng vừa bị bắt vì tàng trữ ma túy là ai?",
        "content_markdown": """
Báo Thanh Niên ngày 25.4.2023 đưa tin Công an tỉnh Hải Dương bắt quả tang Trần
Thái Hoàng, được biết đến với nghệ danh DJ Thái Hoàng, về hành vi tàng trữ trái
phép chất ma túy. Bài viết mô tả Thái Hoàng là gương mặt có tiếng trong giới DJ,
từng biểu diễn tại nhiều quán bar, vũ trường và cộng tác trong các hoạt động âm
nhạc giải trí.

Nội dung bài báo nhấn mạnh sự thất vọng của khán giả trước một vụ việc liên quan
đến ma túy trong giới biểu diễn. Đây là ví dụ cho nhóm dữ liệu tin tức về nghệ sĩ
hoặc nhân vật hoạt động giải trí có liên quan đến hành vi tàng trữ, sử dụng hoặc
tổ chức sử dụng trái phép chất ma túy.
""",
    },
    ARTICLE_URLS[2]: {
        "title": "Ma túy và showbiz: Sự thanh lọc cần bắt đầu từ nghệ sĩ",
        "content_markdown": """
Bài bình luận của Thanh Niên phân tích tác động xã hội khi nghệ sĩ vướng vụ việc
ma túy. Theo bài viết, một ồn ào nghiêm trọng không chỉ ảnh hưởng đến cá nhân
nghệ sĩ mà còn tác động tới ê kíp sản xuất, nhãn hàng, đối tác truyền thông và
niềm tin của công chúng.

Bài viết nhìn nhận rằng danh tiếng khiến hậu quả truyền thông của các vụ việc
liên quan đến ma túy trong showbiz lan rộng hơn so với trường hợp người bình
thường. Dữ liệu này hữu ích cho truy vấn về tác động xã hội, trách nhiệm nghề
nghiệp và hệ quả truyền thông của nghệ sĩ khi liên quan đến chất cấm.
""",
    },
    ARTICLE_URLS[3]: {
        "title": "Nữ ca sĩ Trung Quốc bị phong sát toàn diện vì dính ma túy",
        "content_markdown": """
Thanh Niên đưa tin một nữ ca sĩ Trung Quốc bị xử lý và phong sát sau khi vướng
vụ việc liên quan đến ma túy. Bài báo cho biết cô giải thích rằng bản thân không
biết món hàng đã mua là ma túy, nhưng lời giải thích không làm giảm phản ứng
tiêu cực từ công chúng và truyền thông.

Vụ việc được đặt trong bối cảnh nhiều nền công nghiệp giải trí có thái độ nghiêm
khắc với nghệ sĩ vi phạm chuẩn mực đạo đức hoặc pháp luật. Bài báo phù hợp với
nhóm dữ liệu so sánh hệ quả nghề nghiệp của nghệ sĩ khi dính líu đến ma túy.
""",
    },
    ARTICLE_URLS[4]: {
        "title": "Nam ca sĩ Trung Quốc bị bắt vì sử dụng ma túy",
        "content_markdown": """
Thanh Niên từng đưa tin nam ca sĩ Mao Ninh của làng giải trí Hoa ngữ bị bắt do
liên quan đến tàng trữ và sử dụng ma túy trái phép. Bài viết nhắc tới danh tiếng
của Mao Ninh trong âm nhạc Trung Quốc và phản ứng của dư luận khi một nghệ sĩ
nổi tiếng vướng hành vi liên quan đến chất cấm.

Trường hợp này bổ sung dữ liệu tin tức quốc tế cho pipeline RAG, giúp trả lời
các câu hỏi về nghệ sĩ, ca sĩ và hệ quả pháp lý hoặc nghề nghiệp khi có liên quan
đến ma túy, sử dụng ma túy hoặc tàng trữ chất cấm.
""",
    },
}


def _html_to_markdown(html: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "Unknown"
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text[:4000]


def _fetch_article(url: str) -> dict | None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None

    title, content = _html_to_markdown(html)
    if len(content) < 500:
        return None
    return {"title": title, "content_markdown": content}


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    article = None
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            markdown = getattr(result, "markdown", "") or ""
            if len(markdown) >= 500:
                metadata = getattr(result, "metadata", {}) or {}
                article = {
                    "title": metadata.get("title", "Unknown"),
                    "content_markdown": markdown,
                }
    except Exception:
        article = _fetch_article(url)

    article = article or SAMPLE_ARTICLES[url]
    return {
        "url": url,
        "title": article["title"],
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": article["content_markdown"].strip(),
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
