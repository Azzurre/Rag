from ddgs import DDGS
from web_url_ingest import extract_text_from_url, save_web_article


def search_duckduckgo(query, max_results=10):
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
        return f"{query} site:ufc.com/events OR ESPN UFC next event main card"

    return f"{query} MMA boxing kickboxing Muay Thai grappling fight training technique"


def search_and_extract_web_documents(query, max_results=5):
    search_query = build_search_query(query)

    print(f"Search query: {search_query}")

    results = search_duckduckgo(search_query, max_results=max_results)

    documents = []

    for result in results:
        title = result.get("title", "Unknown search result")
        url = result.get("url")
        snippet = result.get("body", "")

        if not url:
            continue

        # Save the search result snippet as a document too
        if snippet and len(snippet.strip()) > 50:
            documents.append({
                "title": f"Search result: {title}",
                "url": url,
                "text": f"Title: {title}\nURL: {url}\nSnippet: {snippet}"
            })

        try:
            print(f"Extracting: {url}")
            extracted_title, text = extract_text_from_url(url)

            if text and len(text.strip()) > 300:
                save_web_article(url, extracted_title, text)
                print("Saved.")

                documents.append({
                    "title": extracted_title,
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