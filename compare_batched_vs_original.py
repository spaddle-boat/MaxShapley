#!/usr/bin/env python3
"""
Compare Original MaxShapley vs Batched MaxShapley
--------------------------------------------------

This script runs both implementations on the same MuSiQUE queries and compares:
1. Latency (execution time)
2. Token consumption
3. Attribution quality (Jaccard index against ground truth)

Usage:
    python compare_batched_vs_original.py --num_samples 3
    python compare_batched_vs_original.py --num_samples 5 --llm openrouter
"""

import argparse
import time
import json
import numpy as np
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')

from load_data import load_musique_data_sample
from shapley_algorithms.shapley_algos import MaxShapley
from shapley_algorithms.maxshapley_batched import MaxShapleyBatched


def parse_supporting_indices(supporting_facts, context_parts, index):
    """Parse ground truth supporting source indices."""
    supporting_indices = set()
    
    for i, (title, sentences) in enumerate(context_parts):
        for fact_title, _ in supporting_facts:
            if fact_title == title:
                # Load original JSON to verify
                with open('data/musique_annotated_subset.json', 'r') as f:
                    data = json.load(f)
                    json_entry = data[index]
                    
                for ctx in json_entry['context']:
                    if ctx[0] == fact_title and all(x in " ".join(sentences) for x in ctx[1]):
                        supporting_indices.add(i)
                        break
                break
    
    return supporting_indices


def compute_jaccard_index(scores, ground_truth_indices, threshold=0.1):
    """
    Compute Jaccard index between predicted relevant sources and ground truth.
    Sources with score > threshold are considered relevant.
    """
    predicted_relevant = set([i for i, s in enumerate(scores) if s > threshold])
    
    if len(predicted_relevant) == 0 and len(ground_truth_indices) == 0:
        return 1.0
    
    intersection = len(predicted_relevant & ground_truth_indices)
    union = len(predicted_relevant | ground_truth_indices)
    
    return intersection / union if union > 0 else 0.0


def run_comparison(num_samples=3, llm="openrouter"):
    """Run comparison between original and batched implementations."""
    
    print("=" * 80)
    print(" MAXSHAPLEY: ORIGINAL VS BATCHED COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: MuSiQUE")
    print(f"  Samples: {num_samples}")
    print(f"  LLM Provider: {llm}")
    if llm == "openrouter":
        from llm_pipeline import OPENROUTER_MODEL
        print(f"  Model: {OPENROUTER_MODEL}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 80 + "\n")
    
    results = []
    
    for idx in range(num_samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx + 1}/{num_samples}")
        print(f"{'='*80}")
        
        try:
            # Load data
            example = load_musique_data_sample(idx, readable=False)
            sources = []
            for title, sentences in example['context']:
                doc_text = " ".join(sentences)
                sources.append(f"Document '{title}':\n{doc_text}")
            
            question = example['question']
            ground_truth = example.get('answer', '')
            ground_truth_indices = parse_supporting_indices(
                example['supporting_facts'], 
                example['context'],
                idx
            )
            
            print(f"\nQuestion: {question}")
            print(f"Answer: {ground_truth}")
            print(f"Ground truth sources: {sorted(list(ground_truth_indices))}")
            print(f"Number of sources: {len(sources)}")
            
            # Run original MaxShapley
            print(f"\n{'─'*80}")
            print("ORIGINAL MAXSHAPLEY (Individual Scoring)")
            print(f"{'─'*80}")
            
            original = MaxShapley(sources)
            original.zero_out_tokens()
            
            start_time = time.time()
            original_scores = original.compute(
                question=question,
                ground_truth=ground_truth,
                llm=llm
            )
            original_time = time.time() - start_time
            
            original_input_tokens = original.input_tokens
            original_output_tokens = original.output_tokens
            original_jaccard = compute_jaccard_index(original_scores, ground_truth_indices)
            
            print(f"\n✓ Completed in {original_time:.2f}s")
            print(f"  Input tokens: {original_input_tokens}")
            print(f"  Output tokens: {original_output_tokens}")
            print(f"  Total tokens: {original_input_tokens + original_output_tokens}")
            print(f"  Jaccard index: {original_jaccard:.3f}")
            print(f"  Attribution scores: {[f'{s:.3f}' for s in original_scores]}")
            
            # Run batched MaxShapley
            print(f"\n{'─'*80}")
            print("BATCHED MAXSHAPLEY (Batched Scoring)")
            print(f"{'─'*80}")
            
            batched = MaxShapleyBatched(sources)
            batched.zero_out_tokens()
            
            start_time = time.time()
            batched_scores = batched.compute(
                question=question,
                ground_truth=ground_truth,
                llm=llm
            )
            batched_time = time.time() - start_time
            
            batched_input_tokens = batched.input_tokens
            batched_output_tokens = batched.output_tokens
            batched_jaccard = compute_jaccard_index(batched_scores, ground_truth_indices)
            
            print(f"\n✓ Completed in {batched_time:.2f}s")
            print(f"  Input tokens: {batched_input_tokens}")
            print(f"  Output tokens: {batched_output_tokens}")
            print(f"  Total tokens: {batched_input_tokens + batched_output_tokens}")
            print(f"  Jaccard index: {batched_jaccard:.3f}")
            print(f"  Attribution scores: {[f'{s:.3f}' for s in batched_scores]}")
            
            # Compute improvements
            speedup = original_time / batched_time if batched_time > 0 else 0
            token_reduction = 1 - (batched_input_tokens + batched_output_tokens) / (original_input_tokens + original_output_tokens) if (original_input_tokens + original_output_tokens) > 0 else 0
            jaccard_diff = batched_jaccard - original_jaccard
            
            # Compute score correlation (how similar are the attribution scores?)
            score_correlation = np.corrcoef(original_scores, batched_scores)[0, 1] if len(original_scores) > 1 else 1.0
            
            print(f"\n{'─'*80}")
            print("COMPARISON")
            print(f"{'─'*80}")
            print(f"  Speedup: {speedup:.2f}x ({original_time:.1f}s → {batched_time:.1f}s)")
            print(f"  Token reduction: {token_reduction*100:.1f}%")
            print(f"  Jaccard diff: {jaccard_diff:+.3f} ({original_jaccard:.3f} → {batched_jaccard:.3f})")
            print(f"  Score correlation: {score_correlation:.3f}")
            
            # Store results
            results.append({
                'index': idx,
                'question': question,
                'ground_truth_sources': list(ground_truth_indices),
                'original': {
                    'time': original_time,
                    'input_tokens': original_input_tokens,
                    'output_tokens': original_output_tokens,
                    'jaccard': original_jaccard,
                    'scores': original_scores
                },
                'batched': {
                    'time': batched_time,
                    'input_tokens': batched_input_tokens,
                    'output_tokens': batched_output_tokens,
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
            print(f"\n❌ ERROR on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(" SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    if len(results) > 0:
        avg_speedup = np.mean([r['improvements']['speedup'] for r in results])
        avg_token_reduction = np.mean([r['improvements']['token_reduction'] for r in results])
        avg_jaccard_diff = np.mean([r['improvements']['jaccard_diff'] for r in results])
        avg_score_correlation = np.mean([r['improvements']['score_correlation'] for r in results])
        
        avg_original_jaccard = np.mean([r['original']['jaccard'] for r in results])
        avg_batched_jaccard = np.mean([r['batched']['jaccard'] for r in results])
        
        avg_original_time = np.mean([r['original']['time'] for r in results])
        avg_batched_time = np.mean([r['batched']['time'] for r in results])
        
        print(f"Samples completed: {len(results)}/{num_samples}")
        print(f"\nPerformance:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Average time: {avg_original_time:.1f}s → {avg_batched_time:.1f}s")
        print(f"  Average token reduction: {avg_token_reduction*100:.1f}%")
        print(f"\nQuality:")
        print(f"  Original avg Jaccard: {avg_original_jaccard:.3f}")
        print(f"  Batched avg Jaccard: {avg_batched_jaccard:.3f}")
        print(f"  Average Jaccard diff: {avg_jaccard_diff:+.3f}")
        print(f"  Average score correlation: {avg_score_correlation:.3f}")
        
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
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save results to JSON
        output_file = f'results/batched_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'num_samples': num_samples,
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
                    'avg_batched_time': avg_batched_time
                },
                'results': results
            }, f, indent=2)
        print(f"\n✓ Results saved to: {output_file}")
    
    print(f"\n{'='*80}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare original vs batched MaxShapley")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of MuSiQUE samples to test (default: 3)")
    parser.add_argument("--llm", type=str, default="openrouter", choices=["anthropic", "openai", "openrouter"], 
                        help="LLM provider (default: openrouter)")
    
    args = parser.parse_args()
    
    run_comparison(args.num_samples, args.llm)


if __name__ == "__main__":
    main()

