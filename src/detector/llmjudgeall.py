import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json

from openai import OpenAI
from src.detector.prompts.prompt_config import FEWSHOT_EXAMPLES, PROMPT_TEMPLATE

def format_examples(examples):
    """
    Render FEWSHOT_EXAMPLES list as a JSON array string, with each record formatted as:
    {
      "question": "...",
      "reasoning": "...",
      "answer": "Yes" or "No"
    }
    """
    json_examples = []
    for ex in examples:
        json_examples.append({
            "question": ex["question"],
            "reasoning": ex["thinking"],
            "answer": ex["time_sensitive"]
        })
    # Return JSON string with indentation
    return json.dumps(json_examples, ensure_ascii=False, indent=2)

def check_time_sensitivity(question, client, fewshot_examples, prompt_template, temperature=0.0, model_name='qwen'):
    formatted_examples = format_examples(fewshot_examples)
    prompt = prompt_template.format(
        examples=formatted_examples,
        question=question,
    )

    messages = [
        {"role": "system", "content": "You are a factual knowledge assistant."},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )

    return response

