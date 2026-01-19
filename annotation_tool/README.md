# MaxShapley Annotation Tool

A browser-based annotation interface for labeling source relevance in multi-hop question answering datasets.

## Overview

This tool helps annotators rate the relevance of source passages for answering complex questions across three datasets:
- **HotpotQA**: Multi-hop questions requiring reasoning over multiple documents
- **MS MARCO**: Web search questions with passage-level relevance
- **MuSiQue**: Multi-hop questions with compositional reasoning

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit`: Web UI framework
- `filelock`: Concurrent file access management
- Other project dependencies

### 2. Prepare Data

Run the data preparation script to create the 100-sample datasets and convert existing annotations:

```bash
cd annotation_tool
python data_preparation.py
```

This script will:
- Load the existing 30 annotated samples for each dataset
- Create 100-sample files in `data/samples/`
- Convert existing annotations to the new JSONL format in `data/annotations/`

**Note:** Samples 30-99 are currently placeholders. For production use, replace them with 70 new samples randomly selected from the full datasets.

### 3. Run the Annotation Tool

```bash
streamlit run app.py
```

The tool will open in your default browser at `http://localhost:8501`

## Using the Tool

### Login

1. Enter your name in the sidebar (this tracks your annotations)
2. Select a dataset to annotate

### Annotation Workflow

1. **Read the question** displayed at the top
2. **Review each source** in the expandable sections
3. **Rate each source** using the 0-3 relevance scale:
   - **3**: Highly relevant - contains key information
   - **2**: Moderately relevant - supporting context
   - **1**: Weakly relevant - tangential information
   - **0**: Not relevant - no useful information
4. **Save your annotations** when all sources are rated
5. The tool automatically loads the next sample

### Navigation Controls

- **⏮ Previous**: Go back to the previous sample
- **Skip ⏭**: Skip to the next unannotated sample
- **🗑️ Delete**: Delete your annotation for the current sample (only shown if already annotated)
- **💾 Save & Next**: Save annotations and move to the next sample

### Progress Tracking

The sidebar shows:
- **Your annotations**: Number of samples you've completed
- **Overall progress**: Total samples completed across all annotators
- **Breakdown by annotator**: Individual contributions

## Data Formats

### Input: 100-Sample Datasets

Location: `data/samples/{dataset}_100.json`

Format:
```json
[
  {
    "_id": "sample_id",
    "question": "What is the question?",
    "answer": "The answer",
    "context": [
      ["Source Title 1", ["sentence 1", "sentence 2"]],
      ["Source Title 2", ["sentence 1"]]
    ],
    "supporting_facts": [["Source Title 1", 0], ["Source Title 2", 0]]
  }
]
```

### Output: Annotation JSONL Files

Location: `data/annotations/{dataset}_annotations.jsonl`

Format (one JSON object per line):
```json
{"sample_id": "hotpotqa_35", "dataset": "hotpotqa", "annotator": "alice", "timestamp": "2026-01-19T14:30:00", "annotations": {"Source Title 1": 3, "Source Title 2": 2, "Source Title 3": 0}}
```

Each annotation contains:
- `sample_id`: Unique identifier (format: `{dataset}_{index}`)
- `dataset`: Dataset name
- `annotator`: Name entered in the tool
- `timestamp`: ISO format timestamp
- `annotations`: Dictionary mapping source titles to relevance scores (0-3)

## Features

### Concurrent Annotation

Multiple annotators can use the tool simultaneously:
- Each annotator sees only unannotated samples (for their account)
- File locking prevents data corruption
- Annotations are saved atomically
- Each annotator's progress is tracked independently

### Resume Capability

- Close the browser and come back anytime
- Your progress is automatically saved
- Next session picks up where you left off

### Edit and Delete

- View and modify previous annotations
- Delete annotations if needed
- All changes are tracked with timestamps

## File Structure

```
annotation_tool/
├── app.py                      # Main Streamlit UI
├── annotation_backend.py       # Data persistence layer
├── sample_loader.py            # Dataset loading and formatting
├── config.py                   # Configuration constants
├── data_preparation.py         # Initial data setup script
└── README.md                   # This file

data/
├── annotations/                # Annotation storage
│   ├── hotpotqa_annotations.jsonl
│   ├── msmarco_annotations.jsonl
│   └── musique_annotations.jsonl
├── samples/                    # 100-sample datasets
│   ├── hotpotqa_100.json
│   ├── msmarco_100.json
│   └── musique_100.json
└── [existing subset files]

docs/
└── ANNOTATION_GUIDE.md        # Detailed annotation instructions
```

## Monitoring Progress

### Check Overall Progress

In the Streamlit UI sidebar, you'll see:
- Total samples completed out of 70 per dataset
- Progress percentage
- Breakdown by annotator

### Programmatically Check Progress

```python
from annotation_backend import get_progress_stats

# Overall stats for a dataset
stats = get_progress_stats("hotpotqa")
print(f"Completed: {stats['completed']}/{stats['total_target']}")
print(f"By annotator: {stats['by_annotator']}")

# Individual annotator stats
alice_stats = get_progress_stats("hotpotqa", "alice")
print(f"Alice completed: {alice_stats['completed']} samples")
```

### Inspect Annotation Files

```bash
# Count annotations for a dataset
wc -l data/annotations/hotpotqa_annotations.jsonl

# View recent annotations
tail -n 5 data/annotations/hotpotqa_annotations.jsonl

# View all annotations by a specific annotator
grep '"annotator": "alice"' data/annotations/hotpotqa_annotations.jsonl
```

## Troubleshooting

### Tool won't start

**Error:** `streamlit: command not found`

**Solution:** Install streamlit
```bash
pip install streamlit>=1.30.0
```

### Sample files not found

**Error:** `FileNotFoundError: Sample file not found`

**Solution:** Run the data preparation script
```bash
python annotation_tool/data_preparation.py
```

### File lock timeout

**Error:** `TimeoutError: Could not acquire file lock`

**Solution:**
- Another process might be writing to the file
- Wait a few seconds and try again
- If persistent, check for `.lock` files in `data/annotations/` and delete them

### Changes not saving

**Possible causes:**
- Not all sources are annotated (check for warning message)
- File permission issues (check write access to `data/annotations/`)

**Solution:**
- Ensure all sources have a rating before clicking "Save & Next"
- Check file permissions: `ls -la data/annotations/`

### Browser issues

**Problem:** UI looks broken or buttons don't work

**Solution:**
- Try a different browser (Chrome, Firefox, Safari)
- Clear browser cache
- Restart Streamlit: Ctrl+C and run `streamlit run app.py` again

## Advanced Usage

### Customizing the Relevance Scale

Edit `config.py`:
```python
RELEVANCE_SCALE = {
    0: "Not Relevant",
    1: "Weakly Relevant",
    2: "Moderately Relevant",
    3: "Highly Relevant"
}
```

### Adding More Datasets

1. Add dataset to `config.py`:
```python
DATASETS = ["hotpotqa", "msmarco", "musique", "new_dataset"]

DATASET_CONFIGS = {
    "new_dataset": {
        "name": "New Dataset",
        "subset_file": "new_dataset_annotated_subset.json",
        "has_answer": True,
        "id_field": "_id"
    }
}
```

2. Add subset file to `data/`
3. Run `data_preparation.py`

### Exporting Annotations

```python
import json

# Load all annotations for a dataset
with open('data/annotations/hotpotqa_annotations.jsonl') as f:
    annotations = [json.loads(line) for line in f]

# Convert to DataFrame for analysis
import pandas as pd
df = pd.DataFrame(annotations)

# Export to CSV
df.to_csv('hotpotqa_annotations.csv', index=False)
```

## Best Practices

1. **Read the annotation guide first**: See `docs/ANNOTATION_GUIDE.md`
2. **Annotate in sessions**: 10-15 samples at a time to maintain quality
3. **Take breaks**: Avoid fatigue by splitting annotation across multiple days
4. **Be consistent**: Try to apply the same standards throughout
5. **Ask questions**: If unsure about edge cases, discuss with team

## Support

For issues or questions:
1. Check this README and the annotation guide
2. Review troubleshooting section above
3. Check existing annotations for similar cases
4. Contact the project coordinator

## License

Part of the MaxShapley project.
