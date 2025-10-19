# MAXSHAPLEY: Information Attribution via Shapley Values

Official implementation of MAXSHAPLEY from **[MAXSHAPLEY: Towards Incentive-compatible LLM-based Search with Fair Context Attribution]**.

MAXSHAPLEY computes Shapley values to attribute the contribution of information sources in multi-source question answering using a max-based value function with LLM-assessed relevance.

## Installation

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Usage

```python

# List of information sources (strings)
sources = ["Source 1 text...", "Source 2 text...", "Source 3 text..."]

maxshap = MaxShapley(sources)
shapley_values = maxshap.compute(
    question="Your question",
    ground_truth="Expected answer",
    llm="anthropic"  # or "openai"
)
# Returns normalized scores summing to 1.0
```

## How It Works

1. Generate answer from sources
2. Extract key facts and reasoning steps
3. Score each source's relevance to each key point (0.0-1.0)
4. Calculate Shapley values using max-based formula
5. Normalize to sum to 1.0

## Datasets

We release re-annotated subsets of HotPotQA, MuSiQue, and TREC:
- `hotpotqa_annotated_subset.json`
- `musique_annotated_subset.json`
- `msmarco_annotated_subset.json`

Each contains examples with a query, information sources and their titles, and an indication of which information sources are relevant to answering the query. 
HotPotQA & MuSiQue: supporting_facts field contains tuples of (title, _) where the title indicates relevant sources
TREC: supporting_facts field contains tuples of (title, relevance) where relevance is 0 (not relevant) or above 0 (relevant)