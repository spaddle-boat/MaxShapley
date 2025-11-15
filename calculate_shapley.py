"""
calculate_shapley.py
-------------------
Driver script for computing Shapley values using multiple methods on multi-source QA datasets.

Usage:
    python calculate_shapley.py
    python calculate_shapley.py --dataset hotpot --index 5 --llm openai
"""

import argparse
import logging
import sys
import os
import csv
import time
import tempfile
from datetime import datetime
from typing import Set, Dict, Any

from load_data import (
    load_hotpot_data_sample,
    load_msmarco_data_sample,
    load_musique_data_sample
)

from shapley_algorithms.shapley_algos import (
    FullShapley,
    MonteCarloUniform,
    MonteCarloAntithetic,
    LeaveOneOut,
    MaxShapley
)
from shapley_algorithms.kernel_shap import run_kernel_shap
from llm_pipeline import OPENAI_MODEL, ANTHROPIC_MODEL

def run_experiment(dataset, index, csv_path, llm, samples_u, samples_a, log_dir, timestamp):
    # Load data
    example = None
    if dataset == 'hotpot':
        example = load_hotpot_data_sample(index, readable=False)
        dataset_and_index = f"HotPotQA Index: {index}"
    elif dataset == 'musique':
        example = load_musique_data_sample(index, readable=False)
        dataset_and_index = f"MuSiQUE Index: {index}"
    elif dataset == 'msmarco':
        example = load_msmarco_data_sample(index, readable=False)
        dataset_and_index = f"TREC MS MARCO Index: {index}"
    
    # Extract sources
    sources = []
    for title, sentences in example['context']:
        doc_text = " ".join(sentences)
        sources.append(f"Document '{title}':\n{doc_text}")
    
    question = example['question']
    ground_truth = example.get('answer', '')
    num_sources = len(sources)
    
    # Prepare CSV headers
    shapley_methods = ['FullShapley', 'MaxShapley', 'MonteCarloUniform', 'MonteCarloAntithetic', 'LeaveOneOut']
    new_headers = ['index', 'llm_model']
    for method in shapley_methods:
        new_headers += [f'{method}_shapley_{i}' for i in range(num_sources)]
        new_headers += [f'{method}_execution_time', f'{method}_input_tokens', f'{method}_output_tokens']
    new_headers += ['MonteCarloUniform_sample_size', 'MonteCarloAntithetic_sample_size']
    
    # Handle existing CSV file or create new one
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as fin:
            reader = csv.DictReader(fin)
            old_headers = reader.fieldnames or []
            final_headers = old_headers + [h for h in new_headers if h not in old_headers]
            
            # Atomic write with temp file
            dir_ = os.path.dirname(csv_path) or "."
            fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
            os.close(fd)
            try:
                with open(tmp, 'w', newline='') as fout:
                    writer = csv.DictWriter(fout, fieldnames=final_headers, extrasaction="ignore", restval="")
                    writer.writeheader()
                    for row in reader:
                        writer.writerow(row)
                os.replace(tmp, csv_path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
    else:
        final_headers = list(new_headers)
        with open(csv_path, 'w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=final_headers)
            writer.writeheader()
    
    # Initialize CSV row
    new_row = {h: "" for h in final_headers}
    new_row['index'] = index
    new_row['llm_model'] = OPENAI_MODEL if llm == 'openai' else ANTHROPIC_MODEL
    new_row['MonteCarloUniform_sample_size'] = samples_u
    new_row['MonteCarloAntithetic_sample_size'] = samples_a
    
    # Run FullShapley
    baseline_log = os.path.join(log_dir, f"{dataset}_FullShapley_{timestamp}.log")
    logging.basicConfig(filename=baseline_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
    logging.info(dataset_and_index)
    
    start_time = time.time()
    baseline = FullShapley(sources)
    baseline_values = baseline.compute(
        question=question,
        ground_truth=ground_truth,
        llm=llm
    )
    execution_time = time.time() - start_time
    
    new_row['FullShapley_execution_time'] = execution_time
    new_row['FullShapley_input_tokens'] = baseline.input_tokens
    new_row['FullShapley_output_tokens'] = baseline.output_tokens
    for i in range(num_sources):
        new_row[f'FullShapley_shapley_{i}'] = baseline_values[i]

    print(f"FullShapley: {baseline_log}")
    
    # Run MaxShapley
    maxshapley_log = os.path.join(log_dir, f"{dataset}_MaxShapley_{timestamp}.log")
    logging.basicConfig(filename=maxshapley_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
    logging.info(dataset_and_index)
    
    start_time = time.time()
    maxshap = MaxShapley(sources)
    max_values = maxshap.compute(
        question=question,
        ground_truth=ground_truth,
        llm=llm
    )
    execution_time = time.time() - start_time
    
    new_row['MaxShapley_execution_time'] = execution_time
    new_row['MaxShapley_input_tokens'] = maxshap.input_tokens
    new_row['MaxShapley_output_tokens'] = maxshap.output_tokens
    for i in range(num_sources):
        new_row[f'MaxShapley_shapley_{i}'] = max_values[i]

    print(f"MaxShapley: {maxshapley_log}")

    # Run MonteCarloUniform
    mcu_log = os.path.join(log_dir, f"{dataset}_MonteCarloUniform_{timestamp}.log")
    logging.basicConfig(filename=mcu_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
    logging.info(dataset_and_index)
    
    start_time = time.time()
    mc_uniform = MonteCarloUniform(sources)
    mcu_values = mc_uniform.compute(
        question=question,
        ground_truth=ground_truth,
        llm=llm,
        m=samples_u
    )
    execution_time = time.time() - start_time
    
    new_row['MonteCarloUniform_execution_time'] = execution_time
    new_row['MonteCarloUniform_input_tokens'] = mc_uniform.input_tokens
    new_row['MonteCarloUniform_output_tokens'] = mc_uniform.output_tokens
    for i in range(num_sources):
        new_row[f'MonteCarloUniform_shapley_{i}'] = mcu_values[i]

    print(f"MonteCarloUniform: {mcu_log}")
    
    # Run MonteCarloAntithetic
    mca_log = os.path.join(log_dir, f"{dataset}_MonteCarloAntithetic_{timestamp}.log")
    logging.basicConfig(filename=mca_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
    logging.info(dataset_and_index)
    
    start_time = time.time()
    mc_antithetic = MonteCarloAntithetic(sources)
    mca_values = mc_antithetic.compute(
        question=question,
        ground_truth=ground_truth,
        llm=llm,
        m=samples_a
    )
    execution_time = time.time() - start_time
    
    new_row['MonteCarloAntithetic_execution_time'] = execution_time
    new_row['MonteCarloAntithetic_input_tokens'] = mc_antithetic.input_tokens
    new_row['MonteCarloAntithetic_output_tokens'] = mc_antithetic.output_tokens
    for i in range(num_sources):
        new_row[f'MonteCarloAntithetic_shapley_{i}'] = mca_values[i]

    print(f"MonteCarloAntithetic: {mca_log}")
    
    # Run LeaveOneOut
    loo_log = os.path.join(log_dir, f"{dataset}_LeaveOneOut_{timestamp}.log")
    logging.basicConfig(filename=loo_log, filemode='w', level=logging.INFO, format='%(message)s', force=True)
    logging.info(dataset_and_index)
    
    start_time = time.time()
    loo = LeaveOneOut(sources)
    loo_values = loo.compute(
        question=question,
        ground_truth=ground_truth,
        llm=llm
    )
    execution_time = time.time() - start_time
    
    new_row['LeaveOneOut_execution_time'] = execution_time
    new_row['LeaveOneOut_input_tokens'] = loo.input_tokens
    new_row['LeaveOneOut_output_tokens'] = loo.output_tokens
    for i in range(num_sources):
        new_row[f'LeaveOneOut_shapley_{i}'] = loo_values[i]
    
    print(f"LeaveOneOut: {loo_log}")
    
    # Write row to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_headers)
        writer.writerow(new_row)
    
    # Run KernelSHAP (uses MonteCarloUniform log data)    
    print("Starting KernelSHAP")
    try:
        run_kernel_shap(
            log_file=mcu_log,  # Use MonteCarloUniform log file
            source_csv=csv_path,
            output=csv_path,
            alpha=0.001,
            permutations=samples_u
        )
        print(f"KernelSHAP finished.")
    except Exception as e:
        print(f"KernelSHAP failed: {e}")
    
    print(f"\nResults saved to: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Shapley value experiments")
    parser.add_argument('--dataset', type=str, default='musique', choices=['hotpot', 'musique', 'msmarco'])
    parser.add_argument('--index', type=int, default=0, help='Index of data sample (default: 0)')
    parser.add_argument('--llm', type=str, default='openai', choices=['anthropic', 'openai'])
    parser.add_argument('--log', type=str, default='logs/', help='Path to log file or directory (default: logs/)')
    parser.add_argument('--csv', type=str, default='results/', help='Path to CSV file or directory (default: results/)')
    parser.add_argument('--samples_u', type=int, default=1, help='Samples for MonteCarloUniform (default: 1)')
    parser.add_argument('--samples_a', type=int, default=1, help='Samples for MonteCarloAntithetic (default: 1)')
    
    args = parser.parse_args()
    
    # Configure logging - will create separate log files for each method
    log_dir = args.log
    if not log_dir.endswith('/'):
        log_dir = os.path.dirname(log_dir) or 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure CSV output
    csv_path = args.csv
    if os.path.isdir(csv_path) or csv_path.endswith('/'):
        os.makedirs(csv_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(csv_path, f"{args.dataset}_{timestamp}.csv")
    else:
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

    for idx in range(args.index + 1):
        run_experiment(args.dataset, idx, csv_path, args.llm, args.samples_u, args.samples_a, log_dir, timestamp)

if __name__ == "__main__":
    main()