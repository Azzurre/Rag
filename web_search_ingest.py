from ddgs import DDGS
from web_url_ingest import extract_text_from_url, save_web_article


def search_duckduckgo(query, max_results=3):
    results = []

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            results.append({
                "title": result.get("title"),
                "url": result.get("href"),
                "body": result.get("body")
            })

    return results


def main():
    query = input("Search fight knowledge on the web: ").strip()

    search_query = f"{query} MMA boxing kickboxing Muay Thai technique"

    results = search_duckduckgo(search_query)

    for result in results:
        url = result["url"]

        try:
            print(f"Extracting: {url}")
            title, text = extract_text_from_url(url)

            if text.strip():
                save_web_article(url, title, text)
                print("Saved.")

        except Exception as error:
            print(f"Skipped due to error: {error}")

    print("Done. Now run: python src/ingest.py")


if __name__ == "__main__":
    main()