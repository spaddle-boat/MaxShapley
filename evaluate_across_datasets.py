#!/usr/bin/env python3
"""
Multi-Dataset Evaluation: Original vs Batched MaxShapley
---------------------------------------------------------

Evaluates both implementations on MSMarco, MuSiQUE, and HotpotQA datasets.
Measures cost (time + tokens) and fidelity (Jaccard index).

Usage:
    python evaluate_across_datasets.py --samples_per_dataset 2
"""

import argparse
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Set, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

from shapley_algorithms.shapley_algos import MaxShapley
from shapley_algorithms.maxshapley_batched import MaxShapleyBatched


def load_msmarco_sample(index: int) -> Dict:
    """Load a sample from MSMarco dataset."""
    with open('data/msmarco_annotated_subset.json', 'r') as f:
        data = json.load(f)
        sample = data[index]
        
    # Parse supporting facts (doc_id, sentence_index)
    ground_truth_docs = set([doc_id for doc_id, _ in sample['supporting_facts']])
    
    # Build sources from context
    sources = []
    doc_id_to_index = {}
    for idx, (doc_id, sentences) in enumerate(sample['context']):
        doc_text = " ".join(sentences)
        sources.append(f"Document {doc_id}:\n{doc_text}")
        doc_id_to_index[doc_id] = idx
    
    # Map ground truth doc IDs to source indices
    ground_truth_indices = set([doc_id_to_index[doc_id] for doc_id in ground_truth_docs if doc_id in doc_id_to_index])
    
    return {
        'dataset': 'MSMarco',
        'question': sample['question'],
        'answer': '',  # MSMarco doesn't have explicit answers in this format
        'sources': sources,
        'ground_truth_indices': ground_truth_indices,
        'query_id': sample.get('query_id', index)
    }


def load_musique_sample(index: int) -> Dict:
    """Load a sample from MuSiQUE dataset."""
    with open('data/musique_annotated_subset.json', 'r') as f:
        data = json.load(f)
        sample = data[index]
    
    # Parse supporting facts
    supporting_titles = set([title for title, _ in sample['supporting_facts']])
    
    # Build sources from context
    sources = []
    title_to_index = {}
    for idx, (title, sentences) in enumerate(sample['context']):
        doc_text = " ".join(sentences)
        sources.append(f"Document '{title}':\n{doc_text}")
        title_to_index[title] = idx
    
    # Map ground truth titles to source indices
    ground_truth_indices = set([title_to_index[title] for title in supporting_titles if title in title_to_index])
    
    return {
        'dataset': 'MuSiQUE',
        'question': sample['question'],
        'answer': sample.get('answer', ''),
        'sources': sources,
        'ground_truth_indices': ground_truth_indices,
        'query_id': sample.get('_id', index)
    }


def load_hotpotqa_sample(index: int) -> Dict:
    """Load a sample from HotpotQA dataset."""
    with open('data/hotpotqa_annotated_subset.json', 'r') as f:
        data = json.load(f)
        sample = data[index]
    
    # Parse supporting facts
    supporting_titles = set([title for title, _ in sample['supporting_facts']])
    
    # Build sources from context
    sources = []
    title_to_index = {}
    for idx, (title, sentences) in enumerate(sample['context']):
        doc_text = " ".join(sentences)
        sources.append(f"Document '{title}':\n{doc_text}")
        title_to_index[title] = idx
    
    # Map ground truth titles to source indices
    ground_truth_indices = set([title_to_index[title] for title in supporting_titles if title in title_to_index])
    
    return {
        'dataset': 'HotpotQA',
        'question': sample['question'],
        'answer': sample.get('answer', ''),
        'sources': sources,
        'ground_truth_indices': ground_truth_indices,
        'query_id': sample.get('_id', index)
    }


def compute_jaccard_index(scores: List[float], ground_truth_indices: Set[int], threshold: float = 0.1) -> float:
    """Compute Jaccard index between predicted and ground truth sources."""
    predicted_relevant = set([i for i, s in enumerate(scores) if s > threshold])
    
    if len(predicted_relevant) == 0 and len(ground_truth_indices) == 0:
        return 1.0
    
    intersection = len(predicted_relevant & ground_truth_indices)
    union = len(predicted_relevant | ground_truth_indices)
    
    return intersection / union if union > 0 else 0.0


def run_evaluation(samples_per_dataset: int = 2, llm: str = "openrouter"):
    """Run evaluation across all datasets."""
    
    print("=" * 90)
    print(" MULTI-DATASET EVALUATION: ORIGINAL VS BATCHED MAXSHAPLEY")
    print("=" * 90)
    print(f"\nConfiguration:")
    print(f"  Datasets: MSMarco, MuSiQUE, HotpotQA")
    print(f"  Samples per dataset: {samples_per_dataset}")
    print(f"  LLM Provider: {llm}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 90 + "\n")
    
    all_results = []
    dataset_loaders = {
        'MSMarco': load_msmarco_sample,
        'MuSiQUE': load_musique_sample,
        'HotpotQA': load_hotpotqa_sample
    }
    
    for dataset_name, loader_func in dataset_loaders.items():
        print(f"\n{'='*90}")
        print(f" DATASET: {dataset_name}")
        print(f"{'='*90}\n")
        
        for sample_idx in range(samples_per_dataset):
            print(f"\n{'-'*90}")
            print(f" {dataset_name} - Sample {sample_idx + 1}/{samples_per_dataset}")
            print(f"{'-'*90}")
            
            try:
                # Load sample
                sample = loader_func(sample_idx)
                sources = sample['sources']
                question = sample['question']
                answer = sample['answer']
                ground_truth_indices = sample['ground_truth_indices']
                
                print(f"\nQuestion: {question}")
                print(f"Answer: {answer if answer else 'N/A'}")
                print(f"Ground truth sources: {sorted(list(ground_truth_indices))}")
                print(f"Number of sources: {len(sources)}")
                
                # Run original MaxShapley
                print(f"\n{'·'*90}")
                print("ORIGINAL MAXSHAPLEY")
                print(f"{'·'*90}")
                
                original = MaxShapley(sources)
                original.zero_out_tokens()
                
                start_time = time.time()
                original_scores = original.compute(
                    question=question,
                    ground_truth=answer,
                    llm=llm
                )
                original_time = time.time() - start_time
                
                original_input_tokens = original.input_tokens
                original_output_tokens = original.output_tokens
                original_total_tokens = original_input_tokens + original_output_tokens
                original_jaccard = compute_jaccard_index(original_scores, ground_truth_indices)
                
                print(f"✓ Completed in {original_time:.2f}s")
                print(f"  Tokens: {original_input_tokens} in + {original_output_tokens} out = {original_total_tokens} total")
                print(f"  Jaccard: {original_jaccard:.3f}")
                print(f"  Scores: {[f'{s:.3f}' for s in original_scores]}")
                
                # Run batched MaxShapley
                print(f"\n{'·'*90}")
                print("BATCHED MAXSHAPLEY")
                print(f"{'·'*90}")
                
                batched = MaxShapleyBatched(sources)
                batched.zero_out_tokens()
                
                start_time = time.time()
                batched_scores = batched.compute(
                    question=question,
                    ground_truth=answer,
                    llm=llm
                )
                batched_time = time.time() - start_time
                
                batched_input_tokens = batched.input_tokens
                batched_output_tokens = batched.output_tokens
                batched_total_tokens = batched_input_tokens + batched_output_tokens
                batched_jaccard = compute_jaccard_index(batched_scores, ground_truth_indices)
                
                print(f"✓ Completed in {batched_time:.2f}s")
                print(f"  Tokens: {batched_input_tokens} in + {batched_output_tokens} out = {batched_total_tokens} total")
                print(f"  Jaccard: {batched_jaccard:.3f}")
                print(f"  Scores: {[f'{s:.3f}' for s in batched_scores]}")
                
                # Compute improvements
                speedup = original_time / batched_time if batched_time > 0 else 0
                token_reduction = 1 - batched_total_tokens / original_total_tokens if original_total_tokens > 0 else 0
                jaccard_diff = batched_jaccard - original_jaccard
                score_correlation = np.corrcoef(original_scores, batched_scores)[0, 1] if len(original_scores) > 1 else 1.0
                
                print(f"\n{'·'*90}")
                print("COMPARISON")
                print(f"{'·'*90}")
                print(f"  Speedup: {speedup:.2f}x ({original_time:.1f}s → {batched_time:.1f}s)")
                print(f"  Token reduction: {token_reduction*100:.1f}%")
                print(f"  Jaccard diff: {jaccard_diff:+.3f} ({original_jaccard:.3f} → {batched_jaccard:.3f})")
                print(f"  Score correlation: {score_correlation:.3f}")
                
                # Store results
                all_results.append({
                    'dataset': dataset_name,
                    'sample_index': sample_idx,
                    'query_id': sample['query_id'],
                    'question': question,
                    'ground_truth_sources': list(ground_truth_indices),
                    'num_sources': len(sources),
                    'original': {
                        'time': original_time,
                        'input_tokens': original_input_tokens,
                        'output_tokens': original_output_tokens,
                        'total_tokens': original_total_tokens,
                        'jaccard': original_jaccard,
                        'scores': original_scores
                    },
                    'batched': {
                        'time': batched_time,
                        'input_tokens': batched_input_tokens,
                        'output_tokens': batched_output_tokens,
                        'total_tokens': batched_total_tokens,
                        'jaccard': batched_jaccard,
                        'scores': batched_scores
                    },
                    'improvements': {
                        'speedup': speedup,
                        'token_reduction': token_reduction,
                        'jaccard_diff': jaccard_diff,
                        'score_correlation': score_correlation
                    }
                })
                
            except Exception as e:
                print(f"\n❌ ERROR on {dataset_name} sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Print summary
    print(f"\n\n{'='*90}")
    print(" OVERALL SUMMARY")
    print(f"{'='*90}\n")
    
    if len(all_results) > 0:
        # Overall statistics
        avg_speedup = np.mean([r['improvements']['speedup'] for r in all_results])
        avg_token_reduction = np.mean([r['improvements']['token_reduction'] for r in all_results])
        avg_jaccard_diff = np.mean([r['improvements']['jaccard_diff'] for r in all_results])
        avg_score_correlation = np.mean([r['improvements']['score_correlation'] for r in all_results])
        
        avg_original_jaccard = np.mean([r['original']['jaccard'] for r in all_results])
        avg_batched_jaccard = np.mean([r['batched']['jaccard'] for r in all_results])
        
        avg_original_time = np.mean([r['original']['time'] for r in all_results])
        avg_batched_time = np.mean([r['batched']['time'] for r in all_results])
        
        avg_original_tokens = np.mean([r['original']['total_tokens'] for r in all_results])
        avg_batched_tokens = np.mean([r['batched']['total_tokens'] for r in all_results])
        
        print(f"Samples completed: {len(all_results)}")
        print(f"\nOverall Performance:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Average time: {avg_original_time:.1f}s → {avg_batched_time:.1f}s")
        print(f"  Average tokens: {avg_original_tokens:.0f} → {avg_batched_tokens:.0f}")
        print(f"  Average token reduction: {avg_token_reduction*100:.1f}%")
        
        print(f"\nOverall Fidelity:")
        print(f"  Original avg Jaccard: {avg_original_jaccard:.3f}")
        print(f"  Batched avg Jaccard: {avg_batched_jaccard:.3f}")
        print(f"  Average Jaccard diff: {avg_jaccard_diff:+.3f}")
        print(f"  Average score correlation: {avg_score_correlation:.3f}")
        
        # Per-dataset breakdown
        print(f"\nPer-Dataset Breakdown:")
        for dataset_name in ['MSMarco', 'MuSiQUE', 'HotpotQA']:
            dataset_results = [r for r in all_results if r['dataset'] == dataset_name]
            if dataset_results:
                ds_speedup = np.mean([r['improvements']['speedup'] for r in dataset_results])
                ds_token_reduction = np.mean([r['improvements']['token_reduction'] for r in dataset_results])
                ds_jaccard = np.mean([r['batched']['jaccard'] for r in dataset_results])
                print(f"  {dataset_name:12s}: {len(dataset_results)} samples, "
                      f"{ds_speedup:.2f}x faster, {ds_token_reduction*100:.0f}% fewer tokens, "
                      f"Jaccard={ds_jaccard:.3f}")
        
        print(f"\nInterpretation:")
        if abs(avg_jaccard_diff) < 0.05:
            print(f"  ✓ Quality maintained: Jaccard difference < 0.05")
        elif avg_jaccard_diff > 0:
            print(f"  ✓ Quality improved: Batched has higher Jaccard!")
        else:
            print(f"  ⚠ Quality decreased: Jaccard dropped by {abs(avg_jaccard_diff):.3f}")
        
        if avg_score_correlation > 0.9:
            print(f"  ✓ Attribution scores highly correlated (r={avg_score_correlation:.3f})")
        elif avg_score_correlation > 0.7:
            print(f"  ~ Attribution scores moderately correlated (r={avg_score_correlation:.3f})")
        else:
            print(f"  ⚠ Attribution scores weakly correlated (r={avg_score_correlation:.3f})")
        
        # Save results
        import os
        os.makedirs('results', exist_ok=True)
        output_file = f'results/multi_dataset_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'samples_per_dataset': samples_per_dataset,
                    'llm': llm,
                    'timestamp': datetime.now().isoformat()
                },
                'summary': {
                    'avg_speedup': avg_speedup,
                    'avg_token_reduction': avg_token_reduction,
                    'avg_jaccard_diff': avg_jaccard_diff,
                    'avg_score_correlation': avg_score_correlation,
                    'avg_original_jaccard': avg_original_jaccard,
                    'avg_batched_jaccard': avg_batched_jaccard,
                    'avg_original_time': avg_original_time,
                    'avg_batched_time': avg_batched_time,
                    'avg_original_tokens': avg_original_tokens,
                    'avg_batched_tokens': avg_batched_tokens
                },
                'results': all_results
            }, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    print(f"\n{'='*90}\n")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate across multiple datasets")
    parser.add_argument("--samples_per_dataset", type=int, default=2, 
                        help="Number of samples to test per dataset (default: 2)")
    parser.add_argument("--llm", type=str, default="openrouter", 
                        choices=["anthropic", "openai", "openrouter"], 
                        help="LLM provider (default: openrouter)")
    
    args = parser.parse_args()
    
    run_evaluation(args.samples_per_dataset, args.llm)


if __name__ == "__main__":
    main()

