import wikipedia
import re
from urllib.parse import urlparse, unquote
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from googlesearch import search
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests

def extract_wikipedia_title(url):
    try:
        path = urlparse(url).path
        if path.startswith("/wiki/"):
            return unquote(path.split("/wiki/")[1])
    except Exception as e:
        print(f"Could not extract title from {url}: {e}")
    return None

# def get_wikipedia_summary(question: str, metadata: dict = None, max_chars: int = 2000) -> str:
#     tried_titles = []

#     if metadata and "urls" in metadata:
#         for url in metadata["urls"]:
#             if "wikipedia.org" in url:
#                 title = extract_wikipedia_title(url)
#                 if title:
#                     tried_titles.append(title)
#                     try:
#                         page = wikipedia.page(title)
#                         return page.content[:max_chars]
#                     except Exception as e:
#                         print(f"[meta-url] Failed: {title} -> {e}")
    
#     query = question
#     if metadata and "topic" in metadata:
#         query = f"{question} {metadata['topic']}"
    
#     search_results = wikipedia.search(query)
    
#     for title in search_results:
#         if title in tried_titles:
#             continue
#         try:
#             page = wikipedia.page(title)
#             return page.content[:max_chars]
#         except Exception as e:
#             print(f"[search] Failed: {title} -> {e}")
#             continue

#     print(f"No Wikipedia summary found for question: '{question}'")
#     return ""