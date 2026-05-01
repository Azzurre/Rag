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


def build_search_query(query):
    query_lower = query.lower()

    # Current / live UFC queries need schedule/card sources, not technique sources
    if "ufc" in query_lower and (
        "next" in query_lower
        or "upcoming" in query_lower
        or "card" in query_lower
        or "fighting" in query_lower
        or "fight" in query_lower
        or "schedule" in query_lower
        or "tonight" in query_lower
        or "today" in query_lower
        or "this weekend" in query_lower
    ):
        return f"{query} UFC official events schedule fight card"

    # Technique/training queries
    return f"{query} MMA boxing kickboxing Muay Thai grappling fight training technique"


def search_and_extract_web_documents(query, max_results=3):
    search_query = build_search_query(query)

    print(f"Search query: {search_query}")

    results = search_duckduckgo(search_query, max_results=max_results)

    documents = []

    for result in results:
        url = result.get("url")

        if not url:
            continue

        try:
            print(f"Extracting: {url}")
            title, text = extract_text_from_url(url)

            if text and len(text.strip()) > 300:
                save_web_article(url, title, text)
                print("Saved.")

                documents.append({
                    "title": title,
                    "url": url,
                    "text": text
                })

        except Exception as error:
            print(f"Skipped due to error: {error}")

    return documents


def main():
    query = input("Search fight knowledge on the web: ").strip()

    documents = search_and_extract_web_documents(query)

    print(f"Saved {len(documents)} web document(s).")
    print("Done. Now run: python src/ingest.py")


if __name__ == "__main__":
    main()