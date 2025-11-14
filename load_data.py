from ast import Dict, List
import csv
import os
import json
import sys
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
from collections import defaultdict

# Load API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
anthropic_client = Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

def load_hotpot_data_sample(index = 0, readable = True):
    """Load a specific example from the HotPotQA dev set."""
    path = "hotpotqa_annotated_subset.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the specified example
    example = data[index]

    if not readable: # return json format
        return example

    # Format context into readable text
    formatted_context = []
    for title, sentences in example["context"]:
        doc_text = " ".join(sentences)
        formatted_context.append(f"Document '{title}':\n{doc_text}")
    
    return {
        "question": example["question"],
        "context": "\n\n".join(formatted_context),
        "answer": example["answer"],  # Dev set has answers
        "supporting_facts": example["supporting_facts"]
    }

def load_msmarco_data_sample(index = 0, readable = True):
    """Load a specific example from the MS MARCO (TREC passages) annotated dataset."""
    path = "msmarco_annotated_subset.json"
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get the specified example
    example = data[index]

    if not readable: # return json format
        return example

    # Format context into readable text
    formatted_context = []
    for title, sentences in example["context"]:
        doc_text = " ".join(sentences)
        formatted_context.append(f"Passage '{title}':\n{doc_text}")
    
    return {
        "question": example["question"],
        "answer": "",
        "context": "\n\n".join(formatted_context),
        "supporting_facts": example["supporting_facts"]
    }

def load_musique_data_sample(index=0, readable=True):
    """Load a specific example from the Musique gold dataset.
    
    Args:
        index: Index of the example to load
        readable: If True, formats output for easy reading. If False, returns raw data.
    """
    path = "musique_annotated_subset.json"    
    if not os.path.isfile(path):
        raise FileNotFoundError(f"musique_gold_big.json not found at {path}. Run musique_make_gold_dataset first.")
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if index < 0 or index >= len(data):
        raise IndexError(f"Index {index} out of range (0..{len(data)-1})")
    
    example = data[index]
    
    if not readable:
        return example
    
    # Format context into readable text
    formatted_context = []
    for title, sentences in example["context"]:
        doc_text = " ".join(sentences)
        formatted_context.append(f"Document '{title}':\n{doc_text}")
    
    return {
        "question": example["question"],
        "answer": example["answer"],
        "context": "\n\n".join(formatted_context),
        "supporting_facts": example["supporting_facts"]
    }
