"""
shapley.py
----------

This module provides a class-based interface for computing Shapley values for information attribution in multi-source question answering tasks.

Shapley values are a principled way to fairly attribute the value of a prediction (e.g., an answer from an LLM) to the different information sources that contributed to it. This file provides two main approaches:

- MaxShapley: Computes Shapley values using a max-based value function, where the value of a subset is determined by the maximum relevance of its sources to key points in the answer, as determined by an LLM pipeline.

"""

from ast import List
import math, random, copy
from itertools import combinations, permutations
import random
import pandas as pd
import numpy as np
import re, json

# Flexible import for llm_pipeline
try:
    from llm_pipeline import create_llm_pipeline
except ImportError:
    try:
        from src.llm_pipeline import create_llm_pipeline
    except ImportError:
        # If both fail, try relative import
        from .llm_pipeline import create_llm_pipeline

class Shapley:
    """
    Abstract base class for Shapley value computation.

    This class is intended to be subclassed by specific Shapley value computation strategies.
    It stores the dataset (list of information sources) and defines the interface for compute().
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.input_tokens = 0
        self.output_tokens = 0

    def compute(self, *args, **kwargs):
        """
        Abstract method to compute Shapley values for the dataset.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    def normalize_scores(self, scores):
        if len(scores) > 0 and sum(scores) != 0:
            scores = [v / sum(scores) if v > 0.0 else 0.0 for v in scores]
        return scores

    def zero_out_tokens(self):
        self.input_tokens = 0
        self.output_tokens = 0

class MaxShapley(Shapley):
    """
    Computes Shapley values using a max-based value function and key point decomposition.
    Usage:
        maxshap = MaxShapley(dataset)
        shapley_values = maxshap.compute(question, llm="anthropic", method=1)

        See llm_pipeline.py and prompts.json for how dataset should be formatted. 
    """
    def compute(self, question, ground_truth, llm="anthropic", method=1):
        information_sources = self.dataset
        llm_pipeline = create_llm_pipeline(llm)
        
        def run_llm_with_sources(source_docs, question, llm_pipeline, ground_truth):
            
            shuffled_source_docs = copy.deepcopy(source_docs)
            random.shuffle(shuffled_source_docs)
            
            # Separate keypoint generation
            response = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            keypoints = llm_pipeline.run_task(
                "separate_keypoints",
                {
                    "question": question,
                    "answer" : response
                }
            )
            generalized_keypoints = llm_pipeline.run_task(
                "generalize_keypoints",
                {
                    "question": question,
                    "answer" : response,
                    "ground_truth": ground_truth,
                    "keypoints": keypoints['keypoints']
                }
            )
            return response, generalized_keypoints

        def compute_relevance_scores_on_the_fly(source, key_point, llm_pipeline, source_idx, keypoint_idx):
            if not key_point.strip():
                return 0.0

            result = llm_pipeline.run_task(
                "keypoint_relevance_scoring",
                {
                    "keypoint": key_point,
                    "source": source
                }
            )

            # Handle tuple result directly
            if isinstance(result, tuple) and len(result) == 2:
                score, explanation = result
                return score

            assert isinstance(result, str), "result must be a string"

            # Fallback: try to parse as string (should not happen, but for robustness)
            try:
                explanation_match = re.search(r"EXPLANATION:\s*(.*?)(?:\n|SCORE:|$)", result, re.DOTALL)
                score_match = re.search(r"SCORE:\s*([0-9.]+)", result)
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
                score = float(score_match.group(1)) if score_match else 0.0
                return score
            except Exception:
                return 0.0

        def shapley_for_max_single_keypoint(x):
            n = len(x)
            phi = [1 for _ in range(n)]
            for i in range(n):
                phi[i] = x[i]/n
                for j in range(1, i):
                    p = 0.0
                    for k in range(2, j + 2):
                        p_A = 1.0 / n
                        p_B = (k - 1) / (n - 1)
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

        result, generalized_result = run_llm_with_sources(information_sources, question, llm_pipeline, ground_truth)
        key_points = generalized_result['keypoints']
        if not key_points:
            return [0.0] * len(information_sources)
        n_sources = len(information_sources)
        n_key_points = len(key_points)
        relevance_matrix = np.zeros((n_sources, n_key_points))
        for i, source in enumerate(information_sources):
            for j, key_point in enumerate(key_points):
                relevance_matrix[i, j] = compute_relevance_scores_on_the_fly(source, key_point, llm_pipeline, i, j)

        shapley_values = shapley_for_max(relevance_matrix)

        # Normalize before returning
        shapley_values = self.normalize_scores(shapley_values)
        return shapley_values     
