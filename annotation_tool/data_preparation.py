"""
Data preparation script to create 100-sample datasets and convert existing annotations.

This script:
1. Loads the existing 30 annotated samples for each dataset
2. Creates 100-sample files (30 existing + 70 placeholders)
3. Converts existing annotations to the new JSONL format

Note: The 70 additional samples (indices 30-99) are currently placeholders created by
duplicating the first 30 samples. For production use, these should be replaced with
70 new samples randomly sampled from the full datasets:
- HotpotQA: http://curtis.ml.cmu.edu/datasets/hotpot/
- MS MARCO: https://microsoft.github.io/msmarco/
- MuSiQue: https://github.com/StonyBrookNLP/musique
"""

import json
import os
import sys
from datetime import datetime

# Add parent directory to path to import from config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, SAMPLES_DIR, ANNOTATIONS_DIR, DATASETS,
    DATASET_CONFIGS, EXISTING_SAMPLES, TARGET_SAMPLES_PER_DATASET
)
from annotation_backend import save_annotation, ensure_annotations_dir


def load_existing_subset(dataset):
    """Load the existing 30-sample annotated subset."""
    subset_file = DATASET_CONFIGS[dataset]["subset_file"]
    path = os.path.join(DATA_DIR, subset_file)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Subset file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {subset_file}")
    return data


def create_100_sample_dataset(dataset, existing_samples):
    """
    Create a 100-sample dataset file.

    For now, this creates placeholders by duplicating the first 30 samples.
    In production, indices 30-99 should be replaced with new samples from the full dataset.
    """
    # Start with the existing 30 samples
    samples_100 = existing_samples.copy()

    # Add 70 placeholder samples by cycling through the first 30
    num_placeholders = TARGET_SAMPLES_PER_DATASET - EXISTING_SAMPLES
    for i in range(num_placeholders):
        # Cycle through the existing samples
        source_sample = existing_samples[i % EXISTING_SAMPLES].copy()

        # Mark as placeholder in a way that doesn't break the structure
        # We'll add a note field if it doesn't exist
        if "note" not in source_sample:
            source_sample["note"] = f"PLACEHOLDER: Replace with new sample {EXISTING_SAMPLES + i}"

        samples_100.append(source_sample)

    # Save to samples directory
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    output_path = os.path.join(SAMPLES_DIR, f"{dataset}_100.json")

    with open(output_path, 'w') as f:
        json.dump(samples_100, f, indent=2)

    print(f"Created {output_path} with {len(samples_100)} samples")
    print(f"  - Samples 0-29: Original annotated samples")
    print(f"  - Samples 30-99: PLACEHOLDERS (replace with new samples from full dataset)")

    return output_path


def convert_existing_annotations(dataset, existing_samples):
    """
    Convert existing supporting_facts annotations to the new JSONL format.

    The original 30 samples were annotated and aligned by two researchers:
    "mingxun" and "sara". We create identical annotations for both to reflect
    that they agreed on these annotations.

    For HotpotQA and MuSiQue:
        - supporting_facts are binary (either in list or not)
        - Convert to: 3 (highly relevant) if in supporting_facts, 0 (not relevant) otherwise

    For MS MARCO:
        - supporting_facts already contain relevance scores
        - Use those scores directly
    """
    ensure_annotations_dir()
    converted_count = 0

    # The two researchers who annotated and aligned the original 30 samples
    annotators = ["mingxun", "sara"]

    for index, sample in enumerate(existing_samples[:EXISTING_SAMPLES]):
        # Extract source IDs from context
        source_ids = [title for title, _ in sample["context"]]

        # Build annotations dict based on dataset type
        annotations = {}

        if dataset == "msmarco":
            # MS MARCO already has relevance scores in supporting_facts
            # Format: [[source_id, relevance_score], ...]
            for source_id, score in sample["supporting_facts"]:
                annotations[source_id] = score
        else:
            # HotpotQA and MuSiQue: binary supporting facts
            # Format: [[title, sentence_id], ...]
            supporting_titles = {title for title, _ in sample["supporting_facts"]}

            for source_id in source_ids:
                # Mark as 3 (highly relevant) if in supporting_facts, 0 otherwise
                annotations[source_id] = 3 if source_id in supporting_titles else 0

        # Create annotation records for both researchers
        for annotator in annotators:
            annotation_dict = {
                "sample_id": f"{dataset}_{index}",
                "dataset": dataset,
                "annotator": annotator,
                "timestamp": "2026-01-15T00:00:00",  # Original annotation date
                "annotations": annotations
            }

            # Save annotation
            save_annotation(annotation_dict, dataset)
            converted_count += 1

    print(f"Converted {converted_count} annotations for {dataset} (2 annotators x {EXISTING_SAMPLES} samples)")


def main():
    """Main data preparation workflow."""
    print("=" * 60)
    print("MaxShapley Annotation Tool - Data Preparation")
    print("=" * 60)
    print()

    for dataset in DATASETS:
        print(f"\nProcessing {dataset.upper()}...")
        print("-" * 60)

        try:
            # Load existing 30 samples
            existing_samples = load_existing_subset(dataset)

            # Create 100-sample dataset file
            create_100_sample_dataset(dataset, existing_samples)

            # Convert existing annotations to new format
            convert_existing_annotations(dataset, existing_samples)

            print(f"✓ {dataset} completed successfully")

        except Exception as e:
            print(f"✗ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Replace placeholder samples (indices 30-99) with new samples from full datasets")
    print("2. Run the annotation tool: streamlit run annotation_tool/app.py")
    print("3. Start annotating!")
    print()
    print("Generated files:")
    print(f"  - data/samples/*_100.json (100-sample datasets)")
    print(f"  - data/annotations/*_annotations.jsonl (converted annotations)")


if __name__ == "__main__":
    main()
