#!/bin/bash
# Run MaxShapley comparison with OpenRouter API key from demo backend
# Usage: ./run_comparison.sh [num_samples] [script_name]

set -e

# Get the script to run (default: compare_batched_vs_original.py)
SCRIPT_NAME=${2:-compare_batched_vs_original.py}

# Get the number of samples (default: 3)
NUM_SAMPLES=${1:-3}

# Check if .env file exists in demo
if [ -f "../demo/.env" ]; then
    echo "Loading OpenRouter API key from demo/.env..."
    export $(grep OPENROUTER_API_KEY ../demo/.env | xargs)
else
    echo "ERROR: Could not find ../demo/.env"
    echo "Please ensure the OpenRouter API key is configured in the demo."
    exit 1
fi

# Verify the API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY not found in .env file"
    exit 1
fi

echo "API Key loaded: ${OPENROUTER_API_KEY:0:15}..."

# Run the appropriate script
if [ "$SCRIPT_NAME" = "evaluate_across_datasets.py" ]; then
    echo "Running multi-dataset evaluation on $NUM_SAMPLES samples per dataset..."
    echo ""
    python3 evaluate_across_datasets.py --samples_per_dataset $NUM_SAMPLES --llm openrouter 2>&1 | tee multi_dataset_eval.log
    echo ""
    echo "Evaluation complete! Results saved to multi_dataset_eval.log"
else
    echo "Running comparison on $NUM_SAMPLES MuSiQUE samples..."
    echo ""
    python3 compare_batched_vs_original.py --num_samples $NUM_SAMPLES --llm openrouter 2>&1 | tee comparison_run.log
    echo ""
    echo "Comparison complete! Results saved to comparison_run.log"
fi

