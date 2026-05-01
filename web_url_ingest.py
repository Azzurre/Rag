import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


DATA_WEB_FOLDER = "data/web"


def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def clean_filename(text):
    return "".join(
        char for char in text
        if char.isalnum() or char in (" ", "_", "-")
    ).strip()


def extract_text_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else "web_article"

    paragraphs = soup.find_all("p")
    text_parts = []

    for paragraph in paragraphs:
        text = paragraph.get_text(" ", strip=True)
        if text:
            text_parts.append(text)

    article_text = "\n\n".join(text_parts)

    return title, article_text


def save_web_article(url, title, text):
    ensure_folder_exists(DATA_WEB_FOLDER)

    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace("www.", "")

    safe_title = clean_filename(title)[:80]
    filename = f"web_{domain}_{safe_title}.txt"
    file_path = os.path.join(DATA_WEB_FOLDER, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"Title: {title}\n")
        file.write(f"URL: {url}\n")
        file.write(f"Source Type: Web Article\n\n")
        file.write(text)

    return file_path


def main():
    url = input("Paste a fight-related article URL: ").strip()

    print("Extracting article text...")
    title, text = extract_text_from_url(url)

    if not text.strip():
        print("No readable article text found.")
        return

    file_path = save_web_article(url, title, text)

    print(f"Saved web article to: {file_path}")
    print("Now run: python src/ingest.py")


if __name__ == "__main__":
    main()