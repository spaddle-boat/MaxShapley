"""Sample loading and formatting for the annotation tool."""

import json
import os
from config import SAMPLES_DIR, DATASET_CONFIGS


def load_sample(dataset, index):
    """
    Load a specific sample from the 100-sample pool.

    Args:
        dataset: Dataset name (hotpotqa, msmarco, musique)
        index: Sample index (0-99)

    Returns:
        dict: The sample data
    """
    path = os.path.join(SAMPLES_DIR, f"{dataset}_100.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sample file not found: {path}\n"
            f"Please run data_preparation.py first to create the 100-sample datasets."
        )

    with open(path, 'r') as f:
        data = json.load(f)

    if index < 0 or index >= len(data):
        raise IndexError(f"Sample index {index} out of range (0-{len(data)-1})")

    return data[index]


def format_sample_for_ui(sample, dataset, index):
    """
    Convert dataset-specific format to common UI format.

    Args:
        sample: Raw sample from dataset
        dataset: Dataset name
        index: Sample index

    Returns:
        dict: Formatted sample with keys:
            - sample_id: Unique identifier
            - dataset: Dataset name
            - question: The question text
            - answer: The answer (or None for MS MARCO)
            - sources: List of {id, text} dicts
    """
    # Extract sources from context
    # Use unique IDs to handle duplicate titles
    sources = []
    title_counts = {}

    for title, sentences in sample["context"]:
        # Handle both list and string formats
        if isinstance(sentences, list):
            text = " ".join(sentences)
        else:
            text = sentences

        # Make ID unique by appending index if title appears multiple times
        if title in title_counts:
            title_counts[title] += 1
            unique_id = f"{title} [{title_counts[title]}]"
        else:
            title_counts[title] = 0
            unique_id = title

        sources.append({
            "id": unique_id,
            "text": text
        })

    # Build common structure
    config = DATASET_CONFIGS[dataset]
    formatted = {
        "sample_id": f"{dataset}_{index}",
        "dataset": dataset,
        "question": sample["question"],
        "answer": sample.get("answer") if config["has_answer"] else None,
        "sources": sources,
        "original_id": sample.get(config["id_field"])
    }

    return formatted


def get_source_ids(sample):
    """
    Extract source IDs from a sample's context.

    Args:
        sample: Sample data

    Returns:
        list: List of source IDs (titles)
    """
    return [title for title, _ in sample["context"]]
