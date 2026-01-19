"""Configuration constants for the MaxShapley annotation tool."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")

# Datasets
DATASETS = ["hotpotqa", "msmarco", "musique"]
TARGET_SAMPLES_PER_DATASET = 100
EXISTING_SAMPLES = 30  # Already annotated
NEW_SAMPLES = 70  # To be annotated

# Relevance scale
RELEVANCE_SCALE = {
    0: "Not Relevant",
    1: "Weakly Relevant",
    2: "Moderately Relevant",
    3: "Highly Relevant"
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    "hotpotqa": {
        "name": "HotpotQA",
        "subset_file": "hotpotqa_annotated_subset.json",
        "has_answer": True,
        "id_field": "_id"
    },
    "msmarco": {
        "name": "MS MARCO",
        "subset_file": "msmarco_annotated_subset.json",
        "has_answer": False,
        "id_field": "query_id"
    },
    "musique": {
        "name": "MuSiQue",
        "subset_file": "musique_annotated_subset.json",
        "has_answer": True,
        "id_field": "_id"
    }
}
