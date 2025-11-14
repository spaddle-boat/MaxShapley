# MAXSHAPLEY: Information Attribution via Shapley Values

Official implementation of MAXSHAPLEY from **[MAXSHAPLEY: Towards Incentive-compatible LLM-based Search with Fair Context Attribution]**.

## Overview

This repository computes Shapley values to fairly attribute the contribution of information sources in multi-source question answering. We implement multiple Shapley value computation methods including our novel **MaxShapley** approach.

## Installation

```bash
pip install -r requirements.txt
```

Set your API keys:
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Quick Start

Run with defaults (MuSiQue dataset, index 0, Anthropic):
```bash
python calculate_shapley.py
```

Run with custom parameters:
```bash
python calculate_shapley.py --dataset hotpot --index 5 --llm openai
```

## Methods Implemented

- **BaselineShapley**: Exact Shapley values (evaluates all coalitions)
- **MaxShapley**: Our novel max-based approach with key point decomposition
- **MonteCarloUniform**: Monte Carlo approximation with uniform sampling
- **MonteCarloAntithetic**: Monte Carlo with variance reduction
- **LeaveOneOut**: Fast approximation using leave-one-out evaluation
- **KernelSHAP**: Linear regression-based approximation (uses MonteCarloUniform logs)

## Command Line Arguments

```bash
python calculate_shapley.py # Will use default parameters
python calculate_shapley.py --dataset hotpot --shapley_methods FullShapley --index 10 
```
**Options:**
- `--dataset`: Dataset to use (`hotpot`, `musique`, `msmarco`) [default: `musique`]
- `--index`: Data sample index (0-29) [default: `0`]
- `--llm`: LLM provider (`anthropic`, `openai`) [default: `anthropic`]
- `--log`: Directory for log files [default: `logs/`]
- `--csv`: Directory for CSV results [default: `results/`]
- `--samples_u`: Number of samples for MonteCarloUniform [default: `16`]
- `--samples_a`: Number of samples for MonteCarloAntithetic [default: `16`]

## Output

The script generates:

**CSV file** (`results/` directory):
- One row per index with Shapley values. 

**Log files** (`logs/` directory):
- Separate log file for each method
- Contains detailed execution traces and intermediate results

## Datasets

We release re-annotated subsets of HotPotQA, MuSiQue, and MS-MARCO:
- `hotpotqa_annotated_subset.json`
- `musique_annotated_subset.json`
- `msmarco_annotated_subset.json`

Each contains examples with a query, information sources and their titles, and an indication of which information sources are relevant to answering the query. 
HotPotQA & MuSiQue: supporting_facts field contains tuples of (title, _) where the title indicates relevant sources
MS-MARCO: supporting_facts field contains tuples of (title, relevance) where relevance is 0 (not relevant) or above 0 (relevant)

Each sample contains:
- `question`: The query
- `context`: List of (title, sentences) tuples
- `answer`: Ground truth answer
- `supporting_facts`: Relevant sources
