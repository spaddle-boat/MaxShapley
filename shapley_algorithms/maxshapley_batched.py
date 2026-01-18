"""
MaxShapley with Batched Relevance Scoring
-------------------------------------------

This is an optimized version of MaxShapley that uses batched relevance scoring
instead of individual (source, keypoint) calls. This should be 6x faster while
maintaining comparable attribution quality.

Key difference: Instead of n_sources × n_keypoints individual LLM calls,
we make n_sources batched calls (one per source, scoring all keypoints at once).
"""

import numpy as np
import logging
import random
import copy
import json
import re
from typing import List, Dict, Tuple

from shapley_algorithms.shapley_algos import Shapley


class MaxShapleyBatched(Shapley):
    """
    MaxShapley with batched relevance scoring optimization.
    """
    
    def compute(self, question, ground_truth, llm="anthropic"):
        from llm_pipeline import create_llm_pipeline
        
        logging.info("\n[MaxShapley-Batched]")
        information_sources = self.dataset
        llm_pipeline = create_llm_pipeline(llm)
        
        def run_llm_with_sources(source_docs, question, llm_pipeline, ground_truth):
            """Generate answer and extract keypoints (same as original)."""
            shuffled_source_docs = copy.deepcopy(source_docs)
            random.shuffle(shuffled_source_docs)
            
            # Generate response
            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            
            # Separate keypoints
            keypoints, in_tokens, out_tokens = llm_pipeline.run_task(
                "separate_keypoints",
                {
                    "question": question,
                    "answer": response
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            
            # Generalize keypoints
            generalized_keypoints, in_tokens, out_tokens = llm_pipeline.run_task(
                "generalize_keypoints",
                {
                    "question": question,
                    "answer": response,
                    "ground_truth": ground_truth,
                    "keypoints": keypoints['keypoints']
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            
            logging.info(f"Answer and justification: {response}")
            logging.info(f"Separate Keypoints: {keypoints}")
            logging.info(f"Generalized Keypoints: {generalized_keypoints}")
            
            return response, generalized_keypoints

        def compute_relevance_scores_batched(source, keypoints_list, llm_pipeline, source_idx):
            """
            Compute relevance scores for ALL keypoints for one source in a single LLM call.
            This is the key optimization: n_sources calls instead of n_sources × n_keypoints.
            """
            if not keypoints_list or len(keypoints_list) == 0:
                return [0.0] * len(keypoints_list)
            
            # Build numbered list of keypoints
            keypoints_formatted = "\n".join([f"{i+1}. {kp}" for i, kp in enumerate(keypoints_list)])
            
            # Create batched prompt
            prompt_sample = {
                "source": source,
                "keypoints": keypoints_formatted
            }
            
            logging.info(f"Source idx: {source_idx} (batched scoring for {len(keypoints_list)} keypoints)")
            logging.info(f"Prompt sample keys: {prompt_sample.keys()}")
            logging.info(f"Source preview: {source[:100]}...")
            logging.info(f"Keypoints formatted preview: {keypoints_formatted[:200]}...")
            
            # Make single LLM call
            try:
                result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "keypoint_relevance_scoring_batched",
                    prompt_sample
                )
            except Exception as e:
                logging.error(f"Error in LLM call: {e}")
                return [0.0] * len(keypoints_list)
                
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            
            # Ensure result is a string
            if not isinstance(result, str):
                result = str(result) if result is not None else ""
            
            logging.info(f"LLM response type: {type(result)}, length: {len(result)}")
            logging.info(f"LLM response (full): '{result}'")
            
            # Parse JSON response: {"scores": [0.7, 0.3, 1.0, ...]}
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', result)
                if json_match:
                    data = json.loads(json_match.group())
                    scores = data.get("scores", [])
                    if len(scores) >= len(keypoints_list):
                        scores = [min(max(float(s), 0.0), 1.0) for s in scores[:len(keypoints_list)]]
                        logging.info(f"Batched scores: {scores}")
                        return scores
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.warning(f"Failed to parse batched scores from JSON: {e}")
            
            # Fallback: try to parse individual numbers
            try:
                numbers = re.findall(r'(\d+\.?\d*)', result)
                scores = [min(max(float(n), 0.0), 1.0) for n in numbers[:len(keypoints_list)]]
                if len(scores) == len(keypoints_list):
                    logging.info(f"Batched scores (fallback): {scores}")
                    return scores
            except ValueError:
                pass
            
            # If all parsing fails, return zeros
            logging.warning(f"Failed to parse batched scores, returning zeros")
            return [0.0] * len(keypoints_list)

        def shapley_for_max_single_keypoint(x):
            """Compute Shapley values for max-based value function (same as original)."""
            n = len(x)
            phi = [1 for _ in range(n)]
            for i in range(n):
                phi[i] = x[i]/n
                for j in range(1, i):
                    p = 0.0
                    for k in range(2, j + 2):
                        p_A = 1.0 / n
                        p_B = (k - 1) / (n - 1) if n > 1 else 0
                        p_C = 1.0
                        for l in range(1, n - j):
                            numerator = n - k - l + 1
                            denominator = n - l - 1
                            if denominator > 0:
                                p_C *= (numerator / denominator)
                            else:
                                p_C = 0.0
                                break
                        p += (p_A * p_B * p_C)
                    phi[i] += p * (x[i] - x[j])
            return phi

        def shapley_for_max(relevance_matrix):
            """Compute Shapley values across all keypoints (same as original)."""
            n_sources = relevance_matrix.shape[0]
            k_keypoints = relevance_matrix.shape[1]
            shapley_values = np.zeros(n_sources)
            for j in range(k_keypoints):
                scores = relevance_matrix[:, j]
                sorted_indices = np.argsort(scores)
                sorted_scores = scores[sorted_indices]
                sorted_phi = shapley_for_max_single_keypoint(sorted_scores.tolist())
                for orig_idx, sorted_idx in enumerate(sorted_indices):
                    shapley_values[sorted_idx] += sorted_phi[orig_idx]
            return shapley_values.tolist()

        # Main computation
        logging.info(f"Sources: {list(range(len(information_sources)))}")
        result, generalized_result = run_llm_with_sources(information_sources, question, llm_pipeline, ground_truth)
        
        key_points = generalized_result['keypoints']
        if not key_points:
            logging.warning("No key points generated by LLM.")
            return [0.0] * len(information_sources)

        n_sources = len(information_sources)
        n_key_points = len(key_points)
        relevance_matrix = np.zeros((n_sources, n_key_points))
        
        # BATCHED SCORING: One call per source instead of n_sources × n_keypoints calls
        for i, source in enumerate(information_sources):
            scores_for_source = compute_relevance_scores_batched(source, key_points, llm_pipeline, i)
            relevance_matrix[i, :] = scores_for_source

        shapley_values = shapley_for_max(relevance_matrix)

        # Normalize before returning
        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values


