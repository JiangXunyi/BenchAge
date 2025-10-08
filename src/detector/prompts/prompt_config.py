# prompt_config.py
FEWSHOT_EXAMPLES = [
    {
        "question": "What U.S. state produces the most peaches?",
        "thinking": "This is a factual question. The answer depends on agricultural data, which can change over time.",
        "time_sensitive": "Yes"
    },
    {
        "question": "What is higher: The number of people living today, or the number of people who have died in the past?",
        "thinking": "This is a factual question, but the answer is stable and unlikely to change significantly.",
        "time_sensitive": "No"
    },
    {
        "question": "If it's cold outside, what does that tell us about global warming?",
        "thinking": "It does not ask for a verifiable fact or specific data, but rather for an explanation of a complex phenomenon, making it subjective and interpretive with no single quantifiable answer.",
        "time_sensitive": "No"
    },
    {
        "question": "What is an example of a place where people smoke more on average than in Spain?",
        "thinking": "This is a factual comparison of smoking prevalence between Spain and another locale. Smoking rates fluctuate over time because of changing public-health policies, cultural trends, and legislation. To answer correctly, one needs the latest epidemiological data, so it is time-sensitive.",
        "time_sensitive": "Yes"
    }
]



PROMPT_TEMPLATE = """You are a helpful assistant. Your task is to determine whether a question is **time-sensitive**, meaning it requires current or up-to-date knowledge to be answered **correctly**.

A question is considered **time-sensitive** ONLY IF both of the following are true:
1. It has a **verifiable factual answer**, AND
2. That answer can **change over time** due to events, leadership, scientific progress, or changing data.

A question is NOT time-sensitive if:
- It is **subjective** or opinion-based (e.g., “What is the best medicine?”)
- It is **hypothetical** or open-ended (e.g., “If it's cold outside, what does that tell us about global warming?”)
- It has **no specific factual answer** or depends on personal/local context

You must reason in **two steps**:

**Step 1: Reasoning**  
Start with `Reasoning:` and explain whether the question meets both time-sensitivity criteria.

**Step 2: Final Decision**  
Start with `Answer:` and respond only with `Yes` or `No`.

Here are examples:
---
{examples}
---

Now analyze the following:

Question: {question}

Step 1: Reasoning  
Start with `Reasoning:` and explain whether the question meets both time-sensitivity criteria.

Step 2: Final Decision  
Start with `Answer:` and respond only with `Yes` or `No`.

Finally, output your result as a JSON object **only**, following this schema:

{{
  "reasoning": "<the text after Reasoning:>",
  "answer": "<Yes or No>",
  "confidence": "<a number between 0 and 1 indicating how confident you are in your answer>"
}}
"""
