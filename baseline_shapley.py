"""
shapley.py
----------

This module provides a class-based interface for computing Shapley values for information attribution in multi-source question answering tasks.

Each approach is encapsulated in a class that inherits from the abstract Shapley base class. These classes are designed to be instantiated with a dataset (list of source documents), and their compute methods are called with the necessary arguments to perform the attribution calculation.

"""

from ast import List
import math, random, copy
from itertools import combinations, permutations
import random
import pandas as pd
import numpy as np
import logging, re, json

from llm_pipeline import create_llm_pipeline

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
      
class FullShapley(Shapley):
    """
    Computes exact Shapley values using a brute-force approach with judge1 (0-1 scoring).

    For each possible subset of sources, this class uses an LLM to generate an answer
    and then scores the answer's quality using judge1 which provides a score between 0 and 1.
    The Shapley value for each source is computed by averaging its marginal contribution
    across all possible subsets. Final values are normalized to sum to 1.
    """
    
    def compute(self, question, ground_truth, llm="anthropic"):
        logging.info("\n[FullShapley]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0.0] * n
        llm_pipeline = create_llm_pipeline(llm)
        value_cache = {}
    
        def value_function(subset_indices):
            subset_key = tuple(sorted(subset_indices))
            if subset_key in value_cache:
                logging.debug(f"Cache hit for subset {subset_key}")
                return value_cache[subset_key]
            subset_sources = [source_docs[i] for i in subset_indices]
            
            # Shuffle the subset sources to randomize order
            shuffled_source_docs = copy.deepcopy(subset_sources)
            random.shuffle(shuffled_source_docs)
            
            logging.info(f"subset_indices: {subset_indices}")
            logging.info(f"Subset: {subset_key}")

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")

            value_cache[subset_key] = score
            return score

        for i in range(n):
            # affected_tmp = [0 for _ in range(n)]
            for j in range(n):
                subsets_without_i = [s for s in combinations(range(n), j) if i not in s]
                for subset in subsets_without_i:
                    subset_with_i = list(subset) + [i]
                    v_with = value_function(subset_with_i)
                    v_without = value_function(subset)
                    marginal_tmp = v_with - v_without
                    marginal = max(0, marginal_tmp)
                    shapley_values[i] += marginal / (math.comb(n-1, j) * n)

        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values

class MonteCarloUniform(Shapley):
    """
    Computes approximate Shapley values using a Monte Carlo approximation algorithm with uniform sampling.

    This method samples random permutations and computes marginal contributions along each permutation.
    It provides a good balance between computational efficiency and accuracy for large datasets.

    """
    
    def compute(self, question, ground_truth, llm="anthropic", m=1):
        logging.info("\n[MonteCarloUniform]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0] * n
        llm_pipeline = create_llm_pipeline(llm)
        value_cache = {}
    
        def value_function(subset_indices):
            subset_key = tuple(subset_indices)
            if subset_key in value_cache:
                logging.debug(f"Cache hit for subset {subset_key}")
                return value_cache[subset_key]
            subset_sources = [source_docs[i] for i in subset_indices]
            logging.info(f"Subset: {subset_indices}")

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(subset_sources)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")

            value_cache[subset_key] = score
            return score

        seen_perms = set() # to ensure no duplicate permutations
        subset = []
        empty_value = value_function(subset)
        for _ in range(m):
            done = False
            while not done:
                perm = [int(x) for x in np.random.permutation(n)]
                if tuple(perm) not in seen_perms:
                    seen_perms.add(tuple(perm))
                    done = True

            subset = []
            prev_value = empty_value

            for idx in perm:
                subset_with_i = subset + [idx]
                new_value = value_function(subset_with_i)
                marginal_contribution = new_value - prev_value
                shapley_values[idx] += max(0, marginal_contribution) / m
                prev_value = new_value
                subset = subset_with_i
            
            logging.info(f"Token consumption at: {self.input_tokens}, {self.output_tokens}")


        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values

class MonteCarloAntithetic(Shapley):
    """
    Computes approximate Shapley values using a Monte Carlo approximation algorithm with antithetic sampling.

    This method uses antithetic sampling by evaluating both forward and reverse permutations,
    which can reduce variance and improve convergence compared to uniform sampling.
    """
    
    def compute(self, question, ground_truth, llm="anthropic", m=16):
        logging.info("\n[MonteCarloAntithetic]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0.0] * n
        llm_pipeline = create_llm_pipeline(llm)
        value_cache = {}
    
        def value_function(subset_indices):
            subset_key = tuple(subset_indices)
            if subset_key in value_cache:
                logging.info(f"Cache hit for subset {subset_key}")
                return value_cache[subset_key]
            subset_sources = [source_docs[i] for i in subset_indices]
            logging.info(f"Subset: {subset_indices}")

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(subset_sources)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")

            value_cache[subset_key] = score
            return score

        seen_perms = set() # to ensure no duplicate permutations
        empty_value = value_function([])
        for _ in range(m):
            done = False
            while not done:
                perm = [int(x) for x in np.random.permutation(n)]
                reverse_perm = list(reversed(perm))
                if (tuple(perm) not in seen_perms) and (tuple(reverse_perm) not in seen_perms):
                    seen_perms.add(tuple(perm))
                    done = True

            # Forward pass
            subset = []
            prev_value = empty_value
            for i in range(n):
                subset_with_i = subset + [perm[i]]
                new_value = value_function(subset_with_i)
                marginal = max(0, new_value - prev_value)  # Custom clipping
                shapley_values[perm[i]] += marginal / m
                prev_value = new_value
                subset = subset_with_i

            # Reverse (antithetic) pass
            subset = []
            prev_value = empty_value
            for i in range(n):
                subset_with_i = subset + [reverse_perm[i]]
                new_value = value_function(subset_with_i)
                marginal = max(0, new_value - prev_value)
                shapley_values[reverse_perm[i]] += marginal / m
                prev_value = new_value
                subset = subset_with_i

            logging.info(f"Token consumption at: {self.input_tokens}, {self.output_tokens}")

        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values

class LeaveOneOut(Shapley):
    """
    Computes approximate Shapley values using Leave One Out algorithm approximation. 
    """
    def compute(self, question, ground_truth, llm="anthropic"):
        logging.info("\n[Leave-One-Out]")
        source_docs = self.dataset
        n = len(source_docs)
        shapley_values = [0.0] * n
        llm_pipeline = create_llm_pipeline(llm)

        def value_function(subset_indices):
            subset_sources = [source_docs[i] for i in subset_indices]
            logging.info(f"Subset: {subset_indices}")

            shuffled_source_docs = copy.deepcopy(subset_sources)
            random.shuffle(shuffled_source_docs)

            response, in_tokens, out_tokens = llm_pipeline.run_task(
                "generate_response_with_info_subset",
                {
                    "question": question,
                    "sources": "\n\n".join(shuffled_source_docs)
                }
            )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens
            logging.info(f"LLM Response: {response}")

            judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                "judge1",
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "response": response
                }
            )
            if "No justification provided" in judge_result['justification']:
                logging.info(f"Trying judge again.")
                judge_result, in_tokens, out_tokens = llm_pipeline.run_task(
                    "judge1",
                    {
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": response
                    }
                )
            self.input_tokens += in_tokens
            self.output_tokens += out_tokens

            logging.info(f"Full Judge Response: {judge_result}")

            score = judge_result['score']
            justification = judge_result['justification']
            logging.info(f"Score: {score}")
            logging.info(f"Judge Explanation: {justification}")
            return score

        utility_with_all = value_function([e for e in range(n)])

        for i in range(n):
            subset = [int(e) for e in range(n) if e != i]
            utility_without = value_function(subset)
            score = max(0, utility_with_all - utility_without) # Clip negatives as per your use case
            shapley_values[i] += score

        shapley_values = self.normalize_scores(shapley_values)
        logging.info(f"Normalized Shapley values: {shapley_values}")
        return shapley_values
