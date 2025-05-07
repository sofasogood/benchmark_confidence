import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.dataset import load_mmlu
import importlib.util
import sys
from tabulate import tabulate
from colorama import init, Fore, Style

# Initialize colorama
init()

def get_metric_params():
    """Define default parameters for metrics that accept them."""
    return {
        'external_validity': {
            'pred_path': "data/predictions/gpt4_preds.csv",
            'data_path': "data/raw/mmlu_test_sampled_0.02.csv"
        },
        'robustness': {
            'orig_path': "data/predictions/gpt4_preds.csv",
            'pert_path': "data/predictions/gpt4_paraphrase.csv"
        },
        'robustness_multi': {
            'pairs': [
                ("data/predictions/gpt4_preds.csv", "Original"),
                ("data/predictions/gpt4_paraphrase.csv", "Paraphrased"),
                ("data/predictions/gpt4_noise.csv", "Surface noise"),
                ("data/predictions/gpt4_shuffle.csv", "Distractor shuffle")
            ]
        }
    }

def get_score_color(score):
    """Return color based on score."""
    if score == 3:
        return Fore.GREEN
    elif score == 2:
        return Fore.LIGHTGREEN_EX
    elif score == 1:
        return Fore.YELLOW
    else:
        return Fore.RED

def create_confidence_table(results):
    """Create a confidence level table from metric results."""
    # Define metric descriptions and their confidence levels
    metric_info = {
        'construct_validity': {
            'description': 'Construct Validity',
            'levels': {
                3: 'Excellent construct validity (κ ≥ 0.8)',
                2: 'Good construct validity (0.6 ≤ κ < 0.8)',
                1: 'Fair construct validity (0.4 ≤ κ < 0.6)',
                0: 'Poor construct validity (κ < 0.4)'
            },
            'metadata': lambda r: f"κ = {r.get('kappa', 'N/A'):.3f}" if r.get('kappa') is not None else "κ = N/A"
        },
        'coverage': {
            'description': 'Coverage',
            'levels': {
                3: 'Excellent coverage (H/Hmax ≥ 0.90)',
                2: 'Good coverage (0.75 ≤ H/Hmax < 0.90)',
                1: 'Fair coverage (0.50 ≤ H/Hmax < 0.75)',
                0: 'Poor coverage (H/Hmax < 0.50)'
            },
            'metadata': lambda r: f"H/Hmax = {r.get('normalized_score', 'N/A'):.3f}" if r.get('normalized_score') is not None else "H/Hmax = N/A"
        },
        'external_validity': {
            'description': 'External Validity',
            'levels': {
                3: 'Excellent external validity (gap ≤ 2pp)',
                2: 'Good external validity (2pp < gap ≤ 5pp)',
                1: 'Fair external validity (5pp < gap ≤ 10pp)',
                0: 'Poor external validity (gap > 10pp)'
            },
            'metadata': lambda r: f"Gap = {r.get('accuracy_gap', 'N/A'):.1f}pp" if r.get('accuracy_gap') is not None else "Gap = N/A"
        },
        'difficulty_discrimination': {
            'description': 'Difficulty & Discrimination',
            'levels': {
                3: 'Excellent difficulty range (< 5% ceiling/floor)',
                2: 'Good difficulty range (5-10% ceiling/floor)',
                1: 'Fair difficulty range (10-20% ceiling/floor)',
                0: 'Poor difficulty range (> 20% ceiling/floor)'
            },
            'metadata': lambda r: f"{r.get('ceiling_percentage', 'N/A'):.1f}% ceiling, {r.get('floor_percentage', 'N/A'):.1f}% floor" if r.get('ceiling_percentage') is not None and r.get('floor_percentage') is not None else "N/A"
        },
        'robustness': {
            'description': 'Robustness',
            'levels': {
                3: 'Excellent robustness (drop ≤ 2pp)',
                2: 'Good robustness (2pp < drop ≤ 5pp)',
                1: 'Fair robustness (5pp < drop ≤ 10pp)',
                0: 'Poor robustness (drop > 10pp)'
            },
            'metadata': lambda r: f"Drop = {r.get('accuracy_drop', 'N/A'):.1f}pp" if r.get('accuracy_drop') is not None else "Drop = N/A"
        },
        'power_ci': {
            'description': 'Power',
            'levels': {
                3: 'Excellent power (CI width ≤ 2pp)',
                2: 'Good power (2pp < CI width ≤ 5pp)',
                1: 'Fair power (5pp < CI width ≤ 10pp)',
                0: 'Poor power (CI width > 10pp)'
            },
            'metadata': lambda r: f"CI width = {r.get('ci_width', 'N/A'):.1f}pp, Accuracy = {r.get('accuracy', 'N/A'):.1f}%" if r.get('ci_width') is not None and r.get('accuracy') is not None else "N/A"
        }
    }

    # Create the table
    table_data = []
    for metric, info in metric_info.items():
        if metric in results:
            score = results[metric].get('score', 0)
            try:
                metadata = info['metadata'](results[metric])
            except Exception as e:
                print(f"Error formatting metadata for {metric}: {e}")
                metadata = "N/A"
            table_data.append({
                'Metric': info['description'],
                'Score': score,
                'Confidence Level': info['levels'][score],
                'Metadata': metadata
            })

    # Convert to DataFrame
    df = pd.DataFrame(table_data)
    
    # Create pretty table for console output
    pretty_table = []
    for _, row in df.iterrows():
        score_color = get_score_color(row['Score'])
        pretty_table.append([
            row['Metric'],
            f"{score_color}{row['Score']}{Style.RESET_ALL}",
            f"{score_color}{row['Confidence Level']}{Style.RESET_ALL}",
            f"{score_color}{row['Metadata']}{Style.RESET_ALL}"
        ])
    
    # Print the pretty table
    print("\n" + "="*100)
    print(f"{Fore.CYAN}Benchmark Confidence Level Table{Style.RESET_ALL}")
    print("="*100)
    print(tabulate(pretty_table, 
                  headers=['Metric', 'Score', 'Confidence Level', 'Metadata'],
                  tablefmt='simple',
                  colalign=('left', 'center', 'left', 'left')))
    print("="*100 + "\n")
    
    return df

def run_metric(metric_name, params=None):
    """Run a metric and get its results."""
    # Get the path to the metric script
    script_path = os.path.join('src', 'metrics', f'{metric_name}.py')
    
    # Load the module
    spec = importlib.util.spec_from_file_location(metric_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Run the metric and get results
    try:
        if params and hasattr(module.main, '__code__') and module.main.__code__.co_argcount > 0:
            return module.main(**params.get(metric_name, {}))
        return module.main()
    except Exception as e:
        print(f"Error running {metric_name}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run benchmark metrics')
    parser.add_argument('--metrics', type=str, nargs='+',
                      help='Specific metrics to run (default: run all)')
    parser.add_argument('--output-dir', type=str, default='data/metrics_results',
                      help='Directory to save results (default: data/metrics_results)')
    parser.add_argument('--pred-path', type=str, default="data/predictions/gpt4_preds.csv",
                      help='Path to predictions file (default: data/predictions/gpt4_preds.csv)')
    parser.add_argument('--data-path', type=str, default="data/raw/mmlu_test_sampled_0.02.csv",
                      help='Path to dataset file (default: data/raw/mmlu_test_sampled_0.02.csv)')
    parser.add_argument('--pert-path', type=str, default="data/predictions/gpt4_paraphrase.csv",
                      help='Path to perturbed predictions file (default: data/predictions/gpt4_paraphrase.csv)')
    parser.add_argument('--pert-files', type=str, nargs='+',
                      help='Additional perturbation files for robustness_multi (default: gpt4_paraphrase.csv gpt4_noise.csv gpt4_shuffle.csv)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define available metrics
    available_metrics = [
        'construct_validity',
        'coverage',
        'external_validity',
        'difficulty_discrimination',
        'robustness',
        'robustness_multi',
        'power_ci'
    ]
    
    # Determine which metrics to run
    metrics_to_run = args.metrics if args.metrics else available_metrics
    
    # Get default parameters and update with command line args
    params = get_metric_params()
    params['external_validity'].update({
        'pred_path': args.pred_path,
        'data_path': args.data_path
    })
    params['robustness'].update({
        'orig_path': args.pred_path,
        'pert_path': args.pert_path
    })
    
    # Update robustness_multi pairs if pert-files provided
    if args.pert_files:
        pert_names = ["Paraphrased", "Surface noise", "Distractor shuffle"]
        params['robustness_multi']['pairs'] = [(args.pred_path, "Original")] + [
            (f, n) for f, n in zip(args.pert_files, pert_names)
        ]
    
    # Run each metric
    results = {}
    for metric in metrics_to_run:
        if metric not in available_metrics:
            print(f"Warning: Unknown metric '{metric}', skipping...")
            continue
            
        print(f"\nRunning {metric}...")
        try:
            result = run_metric(metric, params)
            if result is not None:
                results[metric] = result
                
                # Save individual metric result
                output_file = os.path.join(args.output_dir, f"{metric}_result.json")
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
        except Exception as e:
            print(f"Error running {metric}: {str(e)}")
            continue
    
    # Save combined results
    combined_output = os.path.join(args.output_dir, "all_metrics_results.json")
    with open(combined_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create and save confidence table
    if results:
        confidence_table = create_confidence_table(results)
        table_output = os.path.join(args.output_dir, "confidence_table.csv")
        confidence_table.to_csv(table_output, index=False)
        print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main() 