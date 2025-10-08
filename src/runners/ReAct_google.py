import json
import pandas as pd
from openai import OpenAI
import os
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import requests
from typing import List
import hashlib
import tqdm
import pickle

os.environ["GOOGLE_CSE_ID"] = "your_cse_id" # https://cse.google.com/cse.js?cx=
os.environ["GOOGLE_API_KEY"] = "your_google_api_key" # https://console.cloud.google.com/apis/credentials
# llm_eval = OpenAI(
#     base_url="http://localhost:9090/v1",
#     api_key="EMPTY"
# )


model = "gpt-4o-mini" # e.g llama-3-1-8b


llm_eval = OpenAI(
    # api_key = key
)

model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")


def extract_correct_answers(row):
    correct = []
    # Add keys from mc1_targets where value == 1
    if isinstance(row['mc1_targets'], dict):
        correct += [k for k, v in row['mc1_targets'].items() if v == 1]
    # Add keys from mc2_targets where value == 1
    if isinstance(row['mc2_targets'], dict):
        correct += [k for k, v in row['mc2_targets'].items() if v == 1]
    return correct



def plan_subgoals(question: str, tokens: dict) -> list[str]:
    system = "You are a reasoning assistant."
    prompt = f"""
Your job is to break down the following question into a sequence of smaller information retrieval questions.

Original Question: "{question}"

Please list 2-4 sub-questions needed to answer this.
Output format:
- Subgoal 1: ...
- Subgoal 2: ...
"""
    completion = llm_eval.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    tokens["input_tokens"] += completion.usage.prompt_tokens
    tokens["output_tokens"] += completion.usage.completion_tokens
    response = completion.choices[0].message.content
    return [line.split(": ", 1)[1] for line in response.split("\n") if line.strip().startswith("-")]


def google_search(query: str, api_key: str, cse_id: str, num: int = 5):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num).execute()
    return res.get("items", [])

# ==== get full web page ====
def get_full_text(url: str, max_chars: int = 5000):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, timeout=5, headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[:max_chars]
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

# ==== main search function ====
def search_and_fetch(query: str, num_results: int = 3):
    results = google_search(query, os.environ["GOOGLE_API_KEY"], os.environ["GOOGLE_CSE_ID"], num=num_results)
    all_text = []

    for i, result in enumerate(results):
        fulltext = get_full_text(result["link"])

        all_text.append(fulltext)

    return results, all_text


def extract_key_info_from_texts(texts: List[str], tokens: dict) -> str:
    prompt = "Given the following documents, extract the most relevant facts and evidence that could help answer a factual question. Do NOT generate an answer. Just list facts."
    for i, text in enumerate(texts):
        prompt += f"Document{i+1}:\n{text}\n"

    completion = llm_eval.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    tokens["input_tokens"] += completion.usage.prompt_tokens
    tokens["output_tokens"] += completion.usage.completion_tokens
    
    return completion.choices[0].message.content

def generate_final_answer(question: str, subgoals: list[str], evidences: list[list[dict]], facts_list: list[str], tokens: dict) -> str:
    evidence_str = ""
    for idx, (subgoal, evidence, fact) in enumerate(zip(subgoals, evidences, facts_list)):
        evidence_texts = "\n".join(f"{e['title']} - {e['snippet']}" for e in evidence)
        # tmp_fact = evidence[0]["fact"]
        evidence_texts += f"Facts: {fact}"
        evidence_str += f"\nSubgoal {idx+1}: {subgoal}\nEvidence:\n{evidence_texts}\n"


    system = "You are a helpful and intelligent assistant."
    prompt = f"""
    Your task is to answer the user's question based on the following evidence gathered from web search.

    Main Question:
    {question}

    The following subgoals were used to break down the question, and for each subgoal, supporting evidence is provided.

    {evidence_str}

    Please use the information from the evidence to answer the main question as accurately and clearly as possible.
    - You may synthesize facts from multiple sources.
    - If the evidence contains conflicting or incomplete information, explain what is uncertain.
    - If the answer requires inference (e.g., numerical comparison, trend analysis), explain your reasoning.
    - If the answer is straightforward and factual, answer directly.

    Final Answer:
    """
    # print("Final answer generation prompt:", prompt)


    completion = llm_eval.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    tokens["input_tokens"] += completion.usage.prompt_tokens
    tokens["output_tokens"] += completion.usage.completion_tokens
    
    return completion.choices[0].message.content.strip(), prompt

def enough_fact(facts, question, llm, tokens: dict):
    fact = [f["facts"] for f in facts]
    snippets_text = [f["snippet_text"] for f in facts]

    prompt = f"""
You are helping evaluate whether the current facts are enough to reasonably answer a given question.

QUESTION:
"{question}"

FACTS:
{fact}

SNIPPETS:
{snippets_text}

INSTRUCTION:
- Determine whether the available information is sufficient to give a **reasonable and helpful answer**, even if the data is **not the most recent**, **precise**, or **complete**.
- If the information provides a reasonable estimate or an approximate range that is directly related to the question, consider it **sufficient**.
- Only consider it **insufficient** if the information is clearly irrelevant, outdated by more than a decade without mention, or off-topic.

Respond in **exactly** one of the following two formats:

If sufficient:
Yes

If not sufficient:
No  
REASON: <why it's not sufficient>  
REVISED QUESTION: <a better follow-up query to help answer the original question>
(Include both REASON and REVISED QUESTION)

IMPORTANT:
Strictly follow one of the two response formats above. Do not include anything else.
"""
    
    completion = llm.chat.completions.create(
        model=model,
        messages=[
            # {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    response = completion.choices[0].message.content.strip()
    
    tokens["input_tokens"] += completion.usage.prompt_tokens
    tokens["output_tokens"] += completion.usage.completion_tokens
    
    if response.lower().startswith("yes"):
        return True, ""
    else:
        return False, response

def clean_url(url: str) -> str:
    """Normalize and hash URL to deduplicate."""
    return hashlib.md5(url.encode()).hexdigest()

def get_created_time(result: dict) -> str:
    """Extract creation time from search result."""
    if "pagemap" in result and "metatags" in result["pagemap"]:
        for tag in result["pagemap"]["metatags"]:
            if "moddate" in tag:
                return tag["moddate"]
            elif 'article:modified_time' in tag:
                return tag['article:modified_time']
    return ""  # Default if no time found

def chain_of_action(question: str, tokens: dict) -> tuple[list[str], list[list[dict]]]:
    steps = plan_subgoals(question, tokens) 
    evidences = []
    facts_list = []
    seen_urls = set()
    all_facts_for_checking = []

    for step in steps:
        # print(f"\n Step: {step}")
        google_search_results, bs_results = search_and_fetch(step, num_results=3)

        # Deduplicate using URLs
        unique_results = []
        unique_texts = []
        for result, text in zip(google_search_results, bs_results):
            url_hash = clean_url(result["link"])
            if url_hash not in seen_urls:
                seen_urls.add(url_hash)
                unique_results.append(result) # google search results
                unique_texts.append(text) # bs results

        facts = extract_key_info_from_texts(unique_texts, tokens)
        snippets = [res['snippet'] for res in unique_results]
        snippet_text = "\n".join(snippets)
        evidence_entry = [
            {
                "title": res['title'], 
                "snippet": res['snippet'], 
                "link": (res['link'], get_created_time(res)), 
            } for res in unique_results
        ]
        evidences.append(evidence_entry)
        facts_list.append(facts)

        # Maintain all facts seen so far for more complete judgment
        all_facts_for_checking.append({"facts": facts, "snippet_text": snippet_text})

        attempts = 0
        previous_queries = {step}

        enough, feedback = enough_fact(all_facts_for_checking, step, llm_eval, tokens)
        while not enough and attempts < 2:
            revised_query = None
            for line in feedback.splitlines():
                if line.lower().startswith("revised question:"):
                    revised_query = line.split(":", 1)[-1].strip()
                    break

            if revised_query is None or revised_query in previous_queries:
                break

            previous_queries.add(revised_query)
            attempts += 1

            google_search_results, bs_results = search_and_fetch(revised_query)

            # Only keep new URLs
            unique_results = []
            unique_texts = []
            for result, text in zip(google_search_results, bs_results):
                url_hash = clean_url(result["link"])
                if url_hash not in seen_urls:
                    seen_urls.add(url_hash)
                    unique_results.append(result)
                    unique_texts.append(text)

            facts = extract_key_info_from_texts(unique_texts, tokens)
            snippets = [res['snippet'] for res in unique_results]
            snippet_text = "\n".join(snippets)
            evidence_entry = [
                {
                    "title": res['title'], 
                    "snippet": res['snippet'], 
                    "link": (res['link'], get_created_time(res)), 
                    # "facts": facts
                } for res in unique_results
            ]
            evidences[-1].extend(evidence_entry)
            facts_list[-1] = facts

            # Append to the cumulative facts for reevaluation
            all_facts_for_checking.append({"facts": facts, "snippet_text": snippet_text})

            enough, feedback = enough_fact(all_facts_for_checking, step, llm_eval, tokens)

    return steps, evidences, facts_list

def main(question: str, timeout=100) -> str:
    tokens = {
        "input_tokens": 0,
        "output_tokens": 0
    }
    steps, evidences, facts_list = chain_of_action(question, tokens)
    final_answer, final_answer_prompt = generate_final_answer(question, steps, evidences, facts_list, tokens)
    links = [d["link"] for item in evidences for d in item]
    print(
        f"The input tokens for this question is: {tokens['input_tokens']}, "
        f"output tokens is: {tokens['output_tokens']}, "
        f"the price is {0.15 * tokens['input_tokens'] * 1e-6 + 0.6 * tokens['output_tokens'] * 1e-6:.6f}"
    )

    return {
        "question": question,
        "search_answer": final_answer, 
        "urls" : links, 
        "search_metadata": 
            {
                "steps": steps, 
                "evidences": evidences, 
                "source": "google_search",
                "attempts": len(steps),
                "tokens": tokens,
                "facts": facts_list,
                "final_answer_prompt": final_answer_prompt
            }
        }

if __name__ == "__main__":
    # Load JSONL
    jsonl_path = "data/Evaluation/humaneval/websearch/time-sensitive-105.jsonl"
    df = pd.read_json(jsonl_path, lines=True)

    results = []
    missing_q = []
    for q in tqdm.tqdm(df["question"]):
        print(q)
        try:
            # steps, evidence = chain_of_action(q)
            # result = generate_final_answer(q, steps, evidence)
            result = main(q)
            results.append(result)
        except:
            results.append("")
            missing_q.append(q)
            raise Exception(f"Error processing question: {q}")


    with open("data/Evaluation/humaneval/websearch/time-sensitive-105-results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # human eval
    human_eval = []
    for res in results:
        human_eval.append(
            {
                "question": res["question"],
                "search_answer": res["search_answer"],
                "fact": res["search_metadata"]["facts"],
                "accurate or not": None,
            }
        )
        
    with open("data/Evaluation/humaneval/websearch/human_eval.json", "w") as f:
        json.dump(human_eval, f, indent=4)
        

