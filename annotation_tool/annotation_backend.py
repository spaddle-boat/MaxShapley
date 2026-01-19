"""Backend for annotation data persistence and retrieval."""

import json
import os
from datetime import datetime
from filelock import FileLock
from config import ANNOTATIONS_DIR, EXISTING_SAMPLES, NEW_SAMPLES, TARGET_SAMPLES_PER_DATASET


def ensure_annotations_dir():
    """Create annotations directory if it doesn't exist."""
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)


def get_annotations_path(dataset):
    """Get the path to the annotations JSONL file for a dataset."""
    ensure_annotations_dir()
    return os.path.join(ANNOTATIONS_DIR, f"{dataset}_annotations.jsonl")


def load_annotations(dataset):
    """
    Load all annotations for a dataset from JSONL file.

    Args:
        dataset: Dataset name (hotpotqa, msmarco, musique)

    Returns:
        list: List of annotation dictionaries
    """
    path = get_annotations_path(dataset)

    if not os.path.exists(path):
        return []

    annotations = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                annotations.append(json.loads(line))

    return annotations


def save_annotation(annotation_dict, dataset):
    """
    Append a new annotation to the JSONL file with file locking.

    Args:
        annotation_dict: Dictionary with keys:
            - sample_id: Unique sample identifier
            - dataset: Dataset name
            - annotator: Annotator name
            - timestamp: ISO format timestamp
            - annotations: Dict mapping source_id -> relevance score (0-3)
            - comment: (optional) Free-text comment for this sample
        dataset: Dataset name

    Raises:
        AssertionError: If required keys are missing
        TimeoutError: If file lock cannot be acquired
    """
    # Validate annotation structure
    required_keys = ["sample_id", "dataset", "annotator", "timestamp", "annotations"]
    assert all(k in annotation_dict for k in required_keys), \
        f"Missing required keys. Expected: {required_keys}, Got: {list(annotation_dict.keys())}"

    # Validate annotations are non-empty
    assert len(annotation_dict["annotations"]) > 0, "Annotations cannot be empty"

    # Validate relevance scores
    for source_id, score in annotation_dict["annotations"].items():
        assert score in [0, 1, 2, 3], f"Invalid relevance score {score} for {source_id}"

    path = get_annotations_path(dataset)
    lock_path = f"{path}.lock"

    # Use file locking to prevent concurrent write conflicts
    lock = FileLock(lock_path, timeout=10)
    with lock:
        with open(path, 'a') as f:
            f.write(json.dumps(annotation_dict) + "\n")


def get_next_sample(annotator, dataset, start_from=0):
    """
    Find the next sample that is not completely annotated by this annotator.

    A sample is considered "not complete" if:
    - It has no annotation from this annotator, OR
    - It has an annotation but not all sources are scored

    Args:
        annotator: Annotator name
        dataset: Dataset name
        start_from: Index to start searching from (default 0)

    Returns:
        int or None: Sample index or None if all samples are complete
    """
    from sample_loader import load_sample

    annotations = load_annotations(dataset)

    # Build a dict of sample_id -> annotation for this annotator
    annotator_annotations = {
        a["sample_id"]: a
        for a in annotations
        if a["annotator"] == annotator
    }

    def is_sample_complete(idx):
        """Check if a sample is completely annotated."""
        sample_id = f"{dataset}_{idx}"

        # No annotation at all
        if sample_id not in annotator_annotations:
            return False

        # Has annotation - check if all sources are scored
        try:
            sample = load_sample(dataset, idx)
            num_sources = len(sample.get("context", []))
            num_annotated = len(annotator_annotations[sample_id].get("annotations", {}))
            return num_annotated >= num_sources
        except Exception:
            # If we can't load the sample, consider it incomplete
            return False

    # Find first incomplete sample starting from start_from
    for i in range(start_from, TARGET_SAMPLES_PER_DATASET):
        if not is_sample_complete(i):
            return i

    # If nothing found after start_from, wrap around and search from beginning
    for i in range(0, start_from):
        if not is_sample_complete(i):
            return i

    # All samples have been completely annotated by this annotator
    return None


def get_progress_stats(dataset, annotator=None):
    """
    Calculate annotation progress statistics.

    Args:
        dataset: Dataset name
        annotator: Optional annotator name for individual stats

    Returns:
        dict: Progress statistics
            If annotator specified: {"annotator": str, "completed": int, "total": int}
            If annotator is None: {
                "total_target": int,
                "completed": int,
                "new_completed": int,
                "by_annotator": {annotator: count, ...}
            }
    """
    annotations = load_annotations(dataset)

    if annotator:
        # Individual annotator stats - count ALL samples this annotator has done
        annotator_annotations = [a for a in annotations if a["annotator"] == annotator]
        total_count = len(annotator_annotations)
        new_count = sum(
            1 for a in annotator_annotations
            if int(a["sample_id"].split("_")[1]) >= EXISTING_SAMPLES
        )
        return {
            "annotator": annotator,
            "completed": total_count,  # Total samples annotated (including original 30)
            "new_completed": new_count,  # Only new samples (30-99)
            "total": TARGET_SAMPLES_PER_DATASET
        }

    # Overall stats
    # Get unique samples (in case multiple annotators annotated the same sample)
    unique_samples = set(a["sample_id"] for a in annotations)
    total_done = len(unique_samples)
    new_samples_done = sum(
        1 for sid in unique_samples
        if int(sid.split("_")[1]) >= EXISTING_SAMPLES
    )

    # Per-annotator breakdown (only new samples for the breakdown)
    annotator_counts = {}
    for a in annotations:
        if int(a["sample_id"].split("_")[1]) >= EXISTING_SAMPLES:  # Only count new samples
            annotator_counts[a["annotator"]] = annotator_counts.get(a["annotator"], 0) + 1

    return {
        "total_target": TARGET_SAMPLES_PER_DATASET,
        "completed": total_done,  # Total unique samples done
        "new_completed": new_samples_done,  # New samples done
        "new_target": NEW_SAMPLES,
        "by_annotator": annotator_counts
    }


def get_annotation_for_sample(dataset, sample_id, annotator):
    """
    Retrieve an existing annotation for a specific sample and annotator.

    Args:
        dataset: Dataset name
        sample_id: Sample identifier
        annotator: Annotator name

    Returns:
        dict or None: Annotation dictionary if found, None otherwise
    """
    annotations = load_annotations(dataset)

    for a in annotations:
        if a["sample_id"] == sample_id and a["annotator"] == annotator:
            return a

    return None


def delete_annotation(dataset, sample_id, annotator):
    """
    Delete an annotation by rewriting the JSONL file without the specified annotation.

    Args:
        dataset: Dataset name
        sample_id: Sample identifier
        annotator: Annotator name

    Returns:
        bool: True if annotation was deleted, False if not found
    """
    path = get_annotations_path(dataset)
    lock_path = f"{path}.lock"

    if not os.path.exists(path):
        return False

    lock = FileLock(lock_path, timeout=10)
    with lock:
        # Read all annotations
        annotations = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(json.loads(line))

        # Filter out the annotation to delete
        filtered = [
            a for a in annotations
            if not (a["sample_id"] == sample_id and a["annotator"] == annotator)
        ]

        # Check if anything was deleted
        if len(filtered) == len(annotations):
            return False

        # Rewrite file
        with open(path, 'w') as f:
            for a in filtered:
                f.write(json.dumps(a) + "\n")

        return True


def get_all_annotations_for_sample(dataset, sample_id):
    """
    Get all annotations for a specific sample from all annotators.

    Args:
        dataset: Dataset name
        sample_id: Sample identifier

    Returns:
        list: List of annotation dictionaries for this sample
    """
    annotations = load_annotations(dataset)
    return [a for a in annotations if a["sample_id"] == sample_id]


def get_annotations_summary(dataset):
    """
    Get a summary of all annotations for a dataset.

    Returns:
        dict: {
            sample_id: {
                "annotators": [list of annotators],
                "annotations": {annotator: {source: score, ...}, ...},
                "has_disagreement": bool,
                "max_disagreement": int (max difference in scores for any source)
            }
        }
    """
    annotations = load_annotations(dataset)

    # Group by sample
    by_sample = {}
    for a in annotations:
        sid = a["sample_id"]
        if sid not in by_sample:
            by_sample[sid] = {
                "annotators": [],
                "annotations": {},
                "has_disagreement": False,
                "max_disagreement": 0
            }
        by_sample[sid]["annotators"].append(a["annotator"])
        by_sample[sid]["annotations"][a["annotator"]] = a["annotations"]

    # Calculate disagreements
    for sid, data in by_sample.items():
        if len(data["annotators"]) > 1:
            # Get all source IDs
            all_sources = set()
            for ann in data["annotations"].values():
                all_sources.update(ann.keys())

            # Check disagreement for each source
            max_diff = 0
            for source in all_sources:
                scores = []
                for annotator, ann in data["annotations"].items():
                    if source in ann:
                        scores.append(ann[source])

                if len(scores) > 1:
                    diff = max(scores) - min(scores)
                    max_diff = max(max_diff, diff)

            data["max_disagreement"] = max_diff
            data["has_disagreement"] = max_diff >= 2  # Significant disagreement

    return by_sample


def get_completion_stats(dataset):
    """
    Get detailed completion statistics for each annotator.

    Returns:
        dict: {
            annotator: {
                "complete": int,      # Samples with all sources annotated
                "partial": int,       # Samples with some but not all sources annotated
                "not_started": int,   # Samples with no annotation
                "total": int
            }
        }
    """
    from sample_loader import load_sample

    annotations = load_annotations(dataset)

    # Group annotations by annotator and sample
    by_annotator = {}
    for a in annotations:
        annotator = a["annotator"]
        if annotator not in by_annotator:
            by_annotator[annotator] = {}
        by_annotator[annotator][a["sample_id"]] = a

    # Calculate stats for each annotator
    stats = {}
    for annotator, sample_annotations in by_annotator.items():
        complete = 0
        partial = 0

        for sample_id, annotation in sample_annotations.items():
            try:
                idx = int(sample_id.split("_")[1])
                sample = load_sample(dataset, idx)
                num_sources = len(sample.get("context", []))
                num_annotated = len(annotation.get("annotations", {}))

                if num_annotated >= num_sources:
                    complete += 1
                else:
                    partial += 1
            except Exception:
                # If we can't load the sample, count as partial
                partial += 1

        not_started = TARGET_SAMPLES_PER_DATASET - complete - partial

        stats[annotator] = {
            "complete": complete,
            "partial": partial,
            "not_started": not_started,
            "total": TARGET_SAMPLES_PER_DATASET
        }

    return stats


def export_annotations(dataset, aggregation="average", min_annotators=1, only_complete=True):
    """
    Export annotations in a format compatible with the original annotated subset files.

    Args:
        dataset: Dataset name
        aggregation: How to aggregate multiple annotators' scores
                    - "average": Average scores (rounded)
                    - "mingxun": Use mingxun's annotations only
                    - "sara": Use sara's annotations only
                    - "first": Use first annotator's scores
        min_annotators: Minimum number of annotators required to include a sample
        only_complete: Only export samples where all sources are annotated

    Returns:
        list: List of samples with relevance_labels field added
    """
    from sample_loader import load_sample, format_sample_for_ui

    annotations = load_annotations(dataset)

    # Group annotations by sample
    by_sample = {}
    for a in annotations:
        sid = a["sample_id"]
        if sid not in by_sample:
            by_sample[sid] = {}
        by_sample[sid][a["annotator"]] = a["annotations"]

    exported = []

    for i in range(TARGET_SAMPLES_PER_DATASET):
        sample_id = f"{dataset}_{i}"

        if sample_id not in by_sample:
            continue

        sample_annotations = by_sample[sample_id]

        # Check minimum annotators
        if len(sample_annotations) < min_annotators:
            continue

        # Load original sample
        try:
            raw_sample = load_sample(dataset, i)
            formatted = format_sample_for_ui(raw_sample, dataset, i)
            num_sources = len(formatted["sources"])
        except Exception:
            continue

        # Check completeness if required
        if only_complete:
            all_complete = all(
                len(ann) >= num_sources
                for ann in sample_annotations.values()
            )
            if not all_complete:
                continue

        # Aggregate scores
        source_ids = [s["id"] for s in formatted["sources"]]
        relevance_labels = []

        for source_id in source_ids:
            if aggregation == "average":
                scores = [
                    ann.get(source_id)
                    for ann in sample_annotations.values()
                    if ann.get(source_id) is not None
                ]
                if scores:
                    avg_score = round(sum(scores) / len(scores))
                    relevance_labels.append([source_id, avg_score])
            elif aggregation in sample_annotations:
                # Use specific annotator
                score = sample_annotations[aggregation].get(source_id)
                if score is not None:
                    relevance_labels.append([source_id, score])
            elif aggregation == "first":
                # Use first available annotator
                for ann in sample_annotations.values():
                    score = ann.get(source_id)
                    if score is not None:
                        relevance_labels.append([source_id, score])
                        break

        # Build export record
        export_record = raw_sample.copy()
        export_record["relevance_labels"] = relevance_labels
        export_record["_annotation_meta"] = {
            "sample_id": sample_id,
            "annotators": list(sample_annotations.keys()),
            "aggregation": aggregation
        }

        exported.append(export_record)

    return exported


def get_inter_annotator_stats(dataset):
    """
    Calculate inter-annotator agreement statistics.

    Returns:
        dict: {
            "total_samples": int,
            "samples_with_multiple_annotators": int,
            "samples_with_disagreement": int,
            "average_disagreement": float,
            "disagreement_samples": [list of sample_ids with significant disagreement]
        }
    """
    summary = get_annotations_summary(dataset)

    total = len(summary)
    multi_annotator = sum(1 for s in summary.values() if len(s["annotators"]) > 1)
    with_disagreement = sum(1 for s in summary.values() if s["has_disagreement"])

    disagreement_samples = [
        sid for sid, data in summary.items()
        if data["has_disagreement"]
    ]

    # Average max disagreement for multi-annotator samples
    disagreements = [s["max_disagreement"] for s in summary.values() if len(s["annotators"]) > 1]
    avg_disagreement = sum(disagreements) / len(disagreements) if disagreements else 0

    return {
        "total_samples": total,
        "samples_with_multiple_annotators": multi_annotator,
        "samples_with_disagreement": with_disagreement,
        "average_disagreement": avg_disagreement,
        "disagreement_samples": disagreement_samples
    }
