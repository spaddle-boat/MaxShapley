import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

METHOD = 'MonteCarloUniform'

def solve_lasso_regression(matrix, alpha=0.01, normalize=True, max_iter=1000, random_state=42):
    """
    Solve a linear regression problem using Lasso regularization.
    
    Parameters:
    -----------
    matrix : list of lists or numpy array
        Each row represents a linear equation where the last element is the target (y)
        and the preceding elements are the features (X).
    
    Returns:
    --------
    dict : Dictionary containing regression results
    """
    matrix = np.array(matrix)

    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("Matrix must be 2-dimensional with at least 2 columns")
    
    X = matrix[:, :-1]  # All columns except the last
    y = matrix[:, -1]   # Last column
    
    scaler = None
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)
        lasso.fit(X, y)
    
    r2_score = lasso.score(X, y)
    
    return {
        'coefficients': lasso.coef_,
        'intercept': lasso.intercept_,
        'r2_score': r2_score,
        'alpha_used': alpha,
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
        'scaler': scaler,
        'model': lasso
    }

class LogFileParser:
    """Parser for log files containing subset evaluation results."""
    
    def __init__(self, log_file_path: str, max_coalitions_per_index: Optional[int] = None):
        self.log_file_path = Path(log_file_path)
        self.max_coalitions_per_index = max_coalitions_per_index
        self.raw_data = []
        self.parsed_entries = []
        
    def parse_log_file(self) -> List[Dict]:
        """
        Parse the log file and extract all entries.
        
        Returns:
        --------
        List[Dict] : List of parsed entries, each containing dataset, index, algorithm, model, and subsets
        """
        with open(self.log_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split by dataset entries (looking for pattern: "{dataset} Index: {idx}")
        dataset_pattern = r'(\w+)\s+Index:\s+(\d+)'
        dataset_matches = list(re.finditer(dataset_pattern, content))
        entries = []
        
        for i, match in enumerate(dataset_matches):
            start_pos = match.start()
            end_pos = dataset_matches[i + 1].start() if i + 1 < len(dataset_matches) else len(content)
            
            entry_content = content[start_pos:end_pos]
            parsed_entry = self._parse_single_entry(entry_content)
            
            if parsed_entry:
                entries.append(parsed_entry)
        
        self.parsed_entries = entries
        return entries
    
    def _parse_single_entry(self, entry_content: str) -> Optional[Dict]:
        """Parse a single entry from the log file."""
        lines = entry_content.strip().split('\n')
        
        if not lines:
            return None
        
        # Parse header information
        header_line = lines[0]
        dataset_match = re.match(r'(\w+)\s+Index:\s+(\d+)', header_line)
        
        if not dataset_match:
            return None
        
        dataset = dataset_match.group(1)
        index = int(dataset_match.group(2))
        
        # Find algorithm and model
        algorithm = None
        model = None
        
        for line in lines[1:5]:  # Check first few lines for algorithm and model
            if line.startswith('[') and line.endswith(']'):
                algorithm = line.strip('[').strip(']')
            elif algorithm and not model and line.strip() and not line.startswith('Subset:'):
                model = line.strip()
                break
        
        # Parse subsets and scores
        subsets_and_scores = self._extract_subsets_and_scores(entry_content, self.max_coalitions_per_index)
        
        return {
            'dataset': dataset,
            'index': index,
            'algorithm': algorithm,
            'model': model,
            'subsets_and_scores': subsets_and_scores
        }
    
    def _extract_subsets_and_scores(self, content: str, max_coalitions: Optional[int] = None) -> List[Tuple[List[int], float]]:
        """Extract subset indices and their corresponding scores."""
        subsets_and_scores = []
        
        # Pattern to find "Subset: [indices]"
        subset_pattern = r'Subset:\s*\[([^\]]*)\]'
        # Pattern to find "Score: {score}"
        score_pattern = r'Score:\s*([\d\.\-e]+)'
        
        subset_matches = list(re.finditer(subset_pattern, content))
        
        # Limit the number of coalitions if specified
        if max_coalitions is not None and len(subset_matches) > max_coalitions:
            subset_matches = subset_matches[:max_coalitions]
            print(f"    Limited to first {max_coalitions} coalitions (found {len(list(re.finditer(subset_pattern, content)))} total)")
        
        for subset_match in subset_matches:
            # Extract subset indices
            indices_str = subset_match.group(1).strip()
            if indices_str:
                try:
                    # Parse comma-separated indices
                    indices = [int(x.strip()) for x in indices_str.split(',') if x.strip()]
                except ValueError:
                    continue
            else:
                indices = []  # Empty subset
            
            # Find the first score after this subset
            search_start = subset_match.end()
            score_match = re.search(score_pattern, content[search_start:])
            
            if score_match:
                try:
                    score = float(score_match.group(1))
                    subsets_and_scores.append((indices, score))
                except ValueError:
                    continue
        
        return subsets_and_scores

class KernelSHAPCalculator:
    """Calculate KernelSHAP values from parsed log entries."""
    
    def __init__(self, alpha=0.001, normalize=False, max_coalitions_per_index: Optional[int] = None):
        self.alpha = alpha
        self.normalize = normalize
        self.max_coalitions_per_index = max_coalitions_per_index
        self.results = []
        
    def calculate_shap_values(self, parsed_entries: List[Dict]) -> Dict:
        """
        Calculate SHAP values for all parsed entries.
        
        Parameters:
        -----------
        parsed_entries : List[Dict]
            Parsed entries from log file
            
        Returns:
        --------
        Dict : Results organized by dataset and index
        """
        results = {}
        for entry in parsed_entries:
            dataset = entry['dataset']
            index = entry['index']
            algorithm = entry['algorithm']
            model = entry['model']
            subsets_and_scores = entry['subsets_and_scores']
            
            if not subsets_and_scores:
                print(f"Warning: No valid subsets found for {dataset} Index {index}")
                continue
            
            # Report coalition constraint if applied
            total_coalitions = len(subsets_and_scores)
            if self.max_coalitions_per_index and total_coalitions >= self.max_coalitions_per_index:
                print(f"  Using {self.max_coalitions_per_index} coalitions (limited from {total_coalitions} found)")
            else:
                print(f"  Processing {total_coalitions} coalitions")
            
            # Convert to matrix format for regression
            try:
                matrix, feature_mapping = self._create_regression_matrix(subsets_and_scores)
                
                if len(matrix) < 2:
                    print(f"Warning: Not enough data points for {dataset} Index {index}")
                    continue
                
                # Calculate SHAP values using Lasso regression
                regression_results = solve_lasso_regression(
                    matrix, 
                    alpha=self.alpha, 
                    normalize=self.normalize
                )
                
                # Store results
                key = f"{dataset}_idx_{index}"
                for i in range(3): # TODO: don't hardcode num of rounds
                    tmp_key = key + f"_{i}"
                    if tmp_key not in results:
                        key = tmp_key
                        break

                # Normalizing and clipping negative shapley values
                shap_list = regression_results['coefficients'].tolist()
                for i in range(len(shap_list)):
                    if shap_list[i] < 0: 
                        shap_list[i] = 0
                if len(shap_list) > 0 and sum(shap_list) != 0:
                    shap_list = [v / sum(shap_list) if v > 0.0 else 0.0 for v in shap_list]

                results[key] = {
                    'dataset': dataset,
                    'index': index,
                    'algorithm': algorithm,
                    'model': model,
                    'feature_mapping': feature_mapping,
                    'shap_values': shap_list,
                    'baseline': regression_results['intercept'],
                    'r2_score': regression_results['r2_score'],
                    'n_coalitions': len(subsets_and_scores),
                    'max_coalitions_limit': self.max_coalitions_per_index,
                    'n_features': len(feature_mapping),
                    'regression_matrix': matrix
                }
                
                print(f"✓ Calculated SHAP values for {dataset} Index {index}")
                
            except Exception as e:
                print(f"Error processing {dataset} Index {index}: {str(e)}")
                continue
        
        self.results.append(results)
        return results
    
    def _create_regression_matrix(self, subsets_and_scores: List[Tuple[List[int], float]]) -> Tuple[List[List], Dict[int, int]]:
        """
        Convert subsets and scores to regression matrix format.
        
        Returns:
        --------
        Tuple[List[List], Dict[int, int]] : (matrix, feature_mapping)
            matrix: List of lists where each row is [feature1, feature2, ..., score]
            feature_mapping: Maps original feature indices to matrix column indices
        """
        # Find all unique feature indices 
        all_features = set()
        for indices, _ in subsets_and_scores:
            all_features.update(indices)
        
        # Create mapping from original indices to matrix columns
        sorted_features = sorted(all_features)
        feature_mapping = {feat: i for i, feat in enumerate(sorted_features)}
        n_features = len(sorted_features)
        
        # Create binary matrix
        matrix = []
        for indices, score in subsets_and_scores:
            row = [0] * n_features  # Initialize with zeros
            
            # Set 1 for present features
            for idx in indices:
                if idx in feature_mapping:
                    row[feature_mapping[idx]] = 1
            
            # Append score as the last element
            row.append(score)
            matrix.append(row)
        
        return matrix, {v: k for k, v in feature_mapping.items()}  # Reverse mapping for output
    

    def export_standard_csv(self, output_path: str, source_csv_path: Optional[str] = None, num = 0):
        """Export results with standard columns from source + new SHAP columns."""
        
        # Define which columns to always copy from source CSV
        STANDARD_COLUMNS = [
            'index',
            'max_sources', 
            'selected_indices', 
            'llm_model', 
            'llm_familiarity', 
            'supporting_indices',
            f'KernelSHAP{num}_input_tokens', 
            f'KernelSHAP{num}_output_tokens',
            f'KernelSHAP{num}_execution_time' # Needed for metrics.py to detect KernelSHAP
        ]
        
        # Load source data if provided
        source_data = None
        if source_csv_path:
            try:
                source_data = pd.read_csv(source_csv_path)
                print(f"Loaded source data from {source_csv_path}")
            except Exception as e:
                print(f"Warning: Could not load source CSV {source_csv_path}: {e}")
        
        # Define all headers
        new_headers = [f'KernelSHAP{num}_shapley_{i}' for i in range(6)]
        all_headers = STANDARD_COLUMNS + new_headers
        
        # Check if output file already exists
        output_path = Path(output_path)
        existing_data = None
        if output_path.exists():
            try:
                existing_data = pd.read_csv(output_path)
                print(f"Found existing output file with {len(existing_data)} rows")
                print(f"Existing columns: {list(existing_data.columns)}")
            except Exception as e:
                print(f"Warning: Could not load existing output file: {e}")
        
        # Collect ALL rows from ALL rounds
        all_rows = []
        for i in range(len(self.results)):
            for key, result in self.results[i].items():
                row_data = {}
                
                # Get standard columns from source CSV
                if source_data is not None:
                    matches = source_data[source_data['index'] == result['index']]
                    iloc_idx = 0

                    if len(matches) > 0:
                        source_row = matches.iloc[iloc_idx]
                        print(source_row)
                        for col in STANDARD_COLUMNS:
                            if '_input_tokens' in col:
                                tmp_tokens = 0
                                for tmp_idx in range(len(matches)):
                                    tmp_tokens += int(matches.iloc[tmp_idx][f'{METHOD}_input_tokens'])
                                row_data[col] = tmp_tokens / len(matches)
                            elif '_output_tokens' in col:
                                tmp_tokens = 0
                                for tmp_idx in range(len(matches)):
                                    tmp_tokens += int(matches.iloc[tmp_idx][f'{METHOD}_output_tokens'])
                                row_data[col] = tmp_tokens / len(matches)
                            elif '_execution_time' in col:
                                tmp_time = 0
                                for tmp_idx in range(len(matches)):
                                    tmp_time += matches.iloc[tmp_idx][f'{METHOD}_execution_time']
                                    print(matches.iloc[tmp_idx][f'{METHOD}_execution_time'])
                                row_data[col] = tmp_time / len(matches)
                            else:
                                row_data[col] = source_row[col] if col in source_row else ''
                    else:
                        for col in STANDARD_COLUMNS:
                            row_data[col] = ''
                else:
                    for col in STANDARD_COLUMNS:
                        row_data[col] = ''
                
                # Add new data
                row_data['index'] = result['index']
                
                # Add SHAP values
                for source_idx in range(6):
                    shap_value = 0.0
                    for col_idx, orig_idx in result['feature_mapping'].items():
                        if orig_idx == source_idx:
                            shap_value = result['shap_values'][col_idx]
                            break
                    
                    row_data[f'KernelSHAP{num}_shapley_{source_idx}'] = shap_value

                # Create row in correct order
                row = [row_data[header] for header in all_headers]
                all_rows.append(row)
        
        # Create DataFrame with ALL new rows
        new_df = pd.DataFrame(all_rows, columns=all_headers)
        
        # Combine with existing data if it exists
        if existing_data is not None:
            # Check if columns match exactly
            if list(existing_data.columns) == list(new_df.columns):
                try:
                    # Reset indices to avoid conflicts
                    existing_data = existing_data.reset_index(drop=True)
                    new_df = new_df.reset_index(drop=True)
                    combined_df = pd.concat([existing_data, new_df], ignore_index=True)
                    print(f"Appended {len(new_df)} new rows to {len(existing_data)} existing rows")
                except Exception as e:
                    print(f"Error concatenating DataFrames: {e}")
                    print("Saving new data only...")
                    combined_df = new_df
            else:
                print(f"Column mismatch!")
                print(f"Existing: {list(existing_data.columns)}")
                print(f"New: {list(new_df.columns)}")
                print("Overwriting with new format...")
                combined_df = new_df
        else:
            combined_df = new_df
            print(f"Created new file with {len(new_df)} rows")
        
        # Save ONCE at the end
        combined_df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")


    def export_standard_csv(self, output_path: str, source_csv_path: Optional[str] = None, num = 0):
        """Export results with standard columns from source + new SHAP columns."""
        
        # Define which columns to always copy from source CSV
        STANDARD_COLUMNS = [
            'index',
            f'KernelSHAP{num}_input_tokens', 
            f'KernelSHAP{num}_output_tokens',
            f'KernelSHAP{num}_execution_time' # Needed for metrics.py to detect KernelSHAP
        ]
        
        # Load source data if provided
        source_data = None
        if source_csv_path:
            try:
                source_data = pd.read_csv(source_csv_path)
                print(f"Loaded source data from {source_csv_path}")
            except Exception as e:
                print(f"Warning: Could not load source CSV {source_csv_path}: {e}")
        
        # Define all headers
        new_headers = [f'KernelSHAP{num}_shapley_{i}' for i in range(6)]
        all_headers = STANDARD_COLUMNS + new_headers
        
        # Check if output file already exists
        output_path = Path(output_path)
        existing_data = None
        if output_path.exists():
            try:
                existing_data = pd.read_csv(output_path)
                print(f"Found existing output file with {len(existing_data)} rows")
                print(f"Existing columns: {list(existing_data.columns)}")
            except Exception as e:
                print(f"Warning: Could not load existing output file: {e}")
        
        all_rows = []
        for i in range(len(self.results)):
            for key, result in self.results[i].items():
                row_data = {}
                
                # Get standard columns from source CSV
                if source_data is not None:
                    matches = source_data[source_data['index'] == result['index']]
                    iloc_idx = 0

                    if len(matches) > 0:
                        source_row = matches.iloc[iloc_idx]
                        for col in STANDARD_COLUMNS:
                            if '_input_tokens' in col:
                                tmp_tokens = 0
                                for tmp_idx in range(len(matches)):
                                    tmp_tokens += int(matches.iloc[tmp_idx][f'{METHOD}_input_tokens'])
                                row_data[col] = tmp_tokens / len(matches)
                            elif '_output_tokens' in col:
                                tmp_tokens = 0
                                for tmp_idx in range(len(matches)):
                                    tmp_tokens += int(matches.iloc[tmp_idx][f'{METHOD}_output_tokens'])
                                row_data[col] = tmp_tokens / len(matches)
                            elif '_execution_time' in col:
                                tmp_time = 0
                                for tmp_idx in range(len(matches)):
                                    tmp_time += matches.iloc[tmp_idx][f'{METHOD}_execution_time']
                                row_data[col] = tmp_time / len(matches)
                            else:
                                row_data[col] = source_row[col] if col in source_row else ''
                    else:
                        for col in STANDARD_COLUMNS:
                            row_data[col] = ''
                else:
                    for col in STANDARD_COLUMNS:
                        row_data[col] = ''
                
                # Add new data
                row_data['index'] = result['index']
                
                # Add SHAP values
                for source_idx in range(6):
                    shap_value = 0.0
                    for col_idx, orig_idx in result['feature_mapping'].items():
                        if orig_idx == source_idx:
                            shap_value = result['shap_values'][col_idx]
                            break
                    
                    row_data[f'KernelSHAP{num}_shapley_{source_idx}'] = shap_value

                
                # Create row in correct order
                row = [row_data[header] for header in all_headers]
                all_rows.append(row)
        
        # Create DataFrame with ALL new rows
        new_df = pd.DataFrame(all_rows, columns=all_headers)

        # Combine with existing data if it exists
        if existing_data is not None:
            try:

                 # Reset index to make sure both DataFrames align cleanly
                existing_data = existing_data.reset_index(drop=True)
                new_df = new_df.reset_index(drop=True)
                
                # Align both DataFrames on 'index'
                combined_df = existing_data.merge(
                    new_df[['index'] + [col for col in new_df.columns if col not in existing_data.columns]],
                    on='index',
                    how='outer'
                )

                print(f"Added {len([c for c in new_df.columns if c not in existing_data.columns])} new columns to {output_path}")

                # Reorder columns so new ones come at the end
                existing_cols = list(existing_data.columns)
                new_cols = [c for c in combined_df.columns if c not in existing_cols]
                combined_df = combined_df[existing_cols + new_cols]

            except Exception as e:
                print(f"Error adding new columns: {e}")
                print("Saving new data only...")
                combined_df = new_df
        else:
            combined_df = new_df
            print(f"Created new file with {len(new_df)} rows")
        
        # Save ONCE at the end
        combined_df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

def run_kernel_shap(log_file, source_csv, output, alpha=0.001, permutations=1):
    coalitions = (permutations * 6) + 1
    output = output
    alpha = alpha
    
    METHOD = f'MonteCarloUniform'

    # Parse log file
    print(f"Parsing log file: {log_file}")
    if coalitions:
        print(f"Coalition limit: {coalitions} per index")
    
    log_parser = LogFileParser(log_file, max_coalitions_per_index=coalitions)
    parsed_entries = log_parser.parse_log_file()
    
    print(f"Found {len(parsed_entries)} entries to process")
    
    # Calculate SHAP values
    calculator = KernelSHAPCalculator(
        alpha=alpha, 
        normalize=False,
        max_coalitions_per_index=coalitions
    )
    
    results = calculator.calculate_shap_values(parsed_entries)
        
    # Export results
    calculator.export_standard_csv(output, source_csv, num=permutations)
    
    print(f"\nProcessing complete! Results saved to {output}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Calculate KernelSHAP values from log files')
    parser.add_argument('log_file', help='Path to the log file')
    parser.add_argument('--output', '-o', default='shap_results.csv', 
                       help='Output file path (default: shap_results.csv)')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='csv',
                       help='Output format (default: json)') # Delete json support. 
    parser.add_argument('--alpha', '-a', type=float, default=0.001,
                       help='Lasso regularization parameter (default: 0.01)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable feature normalization')
    parser.add_argument('--permutations', '-p', type=int, default=None,
                       help='Number of permutations to process from MonteCarlo Uniform log files.')
    parser.add_argument('--source-csv', type=str, default=None,
                   help='Source CSV file to copy standard columns from')
    # parser.add_argument('--rounds', type=int, default=1,
    #             help='How many rounds for each index.')               
    
    args = parser.parse_args()
    source_csv = args.source_csv
    log_file = args.log_file

    coalitions = (args.permutations  * 6) + 1
    output = args.output
    alpha = args.alpha
    
    METHOD = f'MonteCarloUniform'

    # Parse log file
    print(f"Parsing log file: {log_file}")
    if coalitions:
        print(f"Coalition limit: {coalitions} per index")
    
    log_parser = LogFileParser(log_file, max_coalitions_per_index=coalitions)
    parsed_entries = log_parser.parse_log_file()
    
    print(f"Found {len(parsed_entries)} entries to process")
    
    # Calculate SHAP values
    calculator = KernelSHAPCalculator(
        alpha=alpha, 
        normalize=False,
        max_coalitions_per_index=coalitions
    )
    
    results = calculator.calculate_shap_values(parsed_entries)
        
    # Export results
    calculator.export_standard_csv(output, source_csv, num=args.permutations)
    
    print(f"\nProcessing complete! Results saved to {output}")


if __name__ == "__main__":
    main()

