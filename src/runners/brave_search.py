from brave import Brave
import time
import os
os.environ['BRAVE_API_KEY']="your_brave_api_key" 
import requests
import logging
import json
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List
from datetime import datetime
from openai import OpenAI
from langchain_openai import ChatOpenAI

model = "gpt-4o-mini" # e.g llama-3-1-8b

llm_eval = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        # base_url=endpoint
    )

model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

def extract_tables_and_infoboxes(soup):
    # extracting tables and infoboxes to be machine readable
    infoboxes = []
    for infobox in soup.find_all("table", class_="infobox"):
        infoboxes.append(table_to_markdown(infobox))

    tables = []
    for table in soup.find_all("table"):
        if "infobox" not in (table.get("class") or []):
            tables.append(table_to_markdown(table))
    return infoboxes, tables

def table_to_markdown(table):
    rows = []
    for tr in table.find_all("tr"):
        cols = tr.find_all(["th", "td"])
        row = [col.get_text(" ", strip=True) for col in cols]
        rows.append(row)
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join([header, sep, body]) if len(rows) > 1 else header

def search_brave_direct(query: str, num_results: int = 3):
    goggle_url = ""
    api_key = os.environ.get("BRAVE_API_KEY")
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {
        "q": query,
        "count": num_results,
        "goggles_id": goggle_url
    }
    resp = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=10)
    data = resp.json()
    urls = []
    for item in data.get("web", {}).get("results", []):
        url = item.get("url")
        if url:
            urls.append(url)
    time.sleep(2)
    return urls

def get_brave_content(question):
    search_results = []
    found_urls = []
    unwanted_domains = ["github.com", "huggingface.co", "huggingface.com", "news.ycombinator.com"]
    urls = search_brave_direct(question, num_results=10)
    for url in urls:
        if any(domain in url for domain in unwanted_domains):
            continue
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            infoboxes, tables = extract_tables_and_infoboxes(soup)
            for table in soup.find_all("table"):
                table.decompose()
            visible_text = soup.get_text(separator=" ", strip=True)
            page_content = {
                "url": url,
                "infoboxes": infoboxes,
                "tables": tables,
                "text": visible_text
            }
            search_results.append(page_content)
            found_urls.append(url)

            # snippet = soup.get_text(separator=" ", strip=True)
            # if snippet:
            #     search_results.append(snippet)
            #     found_urls.append(url)
            if len(search_results) >= 5:
                break
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            continue
    if search_results:
        return search_results, found_urls # returning the webpage as a list
    return [], found_urls  # returning empty list if no results found
    # if search_results:
    #     return "\n\n".join(search_results), found_urls
    # return None, found_urls

def search_wikipedia_helper(question, timeout=10):
    """
    search_wikipedia function is used to search for wikipedia content using the brave search engine. Used in websearch.py.
    Added retry logic (3x) and network error logging
    """
    network_errors = []
    
    for attempt in range(3):
        try:
            content, urls = get_brave_content_with_retry(question, attempt + 1)
            
            if content:
                wikipedia_content = None
                wikipedia_urls = []
                
                print(f"Found {len(content)} pages from Brave search")
                for page in content:
                    print(f"URL: {page['url']}")
                    if "wikipedia.org" in page["url"]:
                        wikipedia_content = page["text"]
                        wikipedia_urls.append(page["url"])
                        print(f"Found Wikipedia page: {page['url']}")
                
                if wikipedia_content:
                    print(f"Processing Wikipedia content for question: {question}")
                    # Use LLM to extract answer from Wikipedia content
                    llm = llm_eval
                    chunk_size = 30000
                    chunks = [wikipedia_content[i:i+chunk_size] for i in range(0, len(wikipedia_content), chunk_size)]
                    
                    answers = []
                    for idx, chunk in enumerate(chunks):
                        chunk_prompt = (
                            f"Given the following Wikipedia page content chunk, answer the question below as accurately as possible using only the information in the chunk.\n\n"
                            f"Web page content chunk {idx+1}/{len(chunks)}:\n{chunk}\n\n"
                            f"Question: {question}\n"
                            f"If you cannot find the answer in the chunk, reply exactly with 'Not found'.\n"
                            f"Answer:"
                        )
                        resp_llm = llm.invoke(chunk_prompt)
                        chunk_answer = resp_llm.content.strip() if hasattr(resp_llm, "content") else str(resp_llm)
                        print(f"LLM answer for chunk {idx+1}: {chunk_answer[:100]}...")
                        
                        if not chunk_answer.strip().lower().startswith("not found"):
                            answers.append({
                                "answer": chunk_answer,
                                "url": wikipedia_urls[0] if wikipedia_urls else "",
                                "last_modified": None,
                                "chunck_prompt": chunk_prompt
                            })
                    
                    if answers:
                        best = answers[0]
                        print(f"Found answer: {best['answer'][:100]}...")
                        return {
                            "search_answer": best["answer"],
                            "urls": [(best["url"], best["last_modified"])],
                            "search_metadata": {
                                "chunck_context": best["chunck_prompt"],
                                "search_epoch": time.time(),
                                "source": "brave_wikipedia_llm",
                                "attempt": attempt + 1
                            },
                        }
                    else:
                        print("LLM could not find answer in Wikipedia content")
                        return {
                            "search_answer": "Not found",
                            "urls": [(wikipedia_urls[0], None)] if wikipedia_urls else [('', None)],
                            "search_metadata": {
                                "search_epoch": time.time(),
                                "source": "brave_wikipedia_llm",
                                "attempt": attempt + 1
                            },
                        }
                else:
                    print(f"No Wikipedia content found for question: {question}")
            
            # If no Wikipedia content found, return None so websearch.py can try google.
            return None
            
        except requests.exceptions.RequestException as e:
            error_info = {
                "question": question,
                "attempt": attempt + 1,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "engine": "brave"
            }
            network_errors.append(error_info)
            logging.warning(f"Brave search attempt {attempt + 1} failed: {e}")
            
            if attempt < 2:
                time.sleep(2)
                continue
            else:
                # Logging network errors
                log_network_errors(network_errors, error_file="network_errors.jsonl")
                return None
                
        except Exception as e:
            logging.exception(f"Brave search attempt {attempt + 1} failed with unexpected error: {e}")
            if attempt == 2:
                return None
            time.sleep(2)
    
    return None

def search_wikipedia(question, timeout=10, chunk_size=30000):
    wiki_result = search_wikipedia_helper(question, timeout=timeout)
    if not wiki_result:
        return {
            "search_answer": "Not found",
            "urls": [('',None)],
            "search_metadata": {
                "search_epoch": time.time(),
                "source": "brave_wikipedia_llm",
                "attempt": 1
            },
        }
    
    # search_wikipedia_helper already processed the content and returned an answer
    if "search_answer" in wiki_result:
        return wiki_result
    
    # fallback
    return {
        "search_answer": "Not found",
        "urls": [('',None)],
        "search_metadata": {
            "search_epoch": time.time(),
            "source": "brave_wikipedia_llm",
            "attempt": wiki_result.get("attempt", 1) if wiki_result else 1
        },
    }

def get_brave_content_with_retry(question: str, attempt: int = 1) -> tuple[List[Dict], List[str]]:
    """
    Enhanced version of get_brave_content with retry logic
    """
    search_results = []
    found_urls = []
    unwanted_domains = ["github.com", "huggingface.co", "huggingface.com", "news.ycombinator.com"]
    
    try:
        urls = search_brave_direct(question, num_results=10)
        
        for url in urls:
            if any(domain in url for domain in unwanted_domains):
                continue
                
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()  # exception for bad status codes
                
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

                infoboxes, tables = extract_tables_and_infoboxes(soup)
                for table in soup.find_all("table"):
                    table.decompose()
                    
                visible_text = soup.get_text(separator=" ", strip=True)
                last_modified = resp.headers.get('last-modified')
                
                page_content = {
                    "url": url,
                    "infoboxes": infoboxes,
                    "tables": tables,
                    "text": visible_text,
                    "last_modified": last_modified,
                    "status_code": resp.status_code
                }
                search_results.append(page_content)
                found_urls.append(url)

                if len(search_results) >= 5:
                    break
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to fetch {url} on attempt {attempt}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Brave search failed on attempt {attempt}: {e}")
        raise
    
    return search_results, found_urls

def log_network_errors(errors, error_file="network_errors.jsonl"):
    try:
        with open(error_file, "a", encoding="utf-8") as f:
            for error in errors:
                json.dump(error, f, ensure_ascii=False)
                f.write("\n")
        logging.info(f"Logged {len(errors)} network errors to {error_file}")
    except Exception as e:
        logging.error(f"Failed to log network errors: {e}")

if __name__ == "__main__":
    # Example usage
    question = "How long should you wait before filing a missing person report?"
    content, urls = get_brave_content_with_retry(question)
    if content:
        first_page = content[0]
        print("URL:", first_page["url"])
        print(first_page["text"])
        # print("\nTABLES:\n")
        # for table in first_page["text"]:
        #     print(table)
        #     print("\n" + "="*40 + "\n")
    else:
        print("No content found.")

    # content = get_brave_content(question)
    # print(content)