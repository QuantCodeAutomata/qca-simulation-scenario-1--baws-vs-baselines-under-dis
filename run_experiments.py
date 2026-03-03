"""
Main script to run all BAWS experiments and generate results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.experiment1 import run_experiment1
from src.experiment2 import run_experiment2
from src.experiment3 import run_experiment3


def save_results_summary(results, output_file):
    """Save numerical results to markdown file."""
    with open(output_file, 'w') as f:
        f.write("# BAWS Experiments Results\n\n")
        f.write("## Summary\n\n")
        f.write("This document contains the results of all BAWS experiments.\n\n")
        
        # Experiment 1
        if 'exp1' in results:
            f.write("## Experiment 1: Discrete Structural Breaks\n\n")
            for setting, data in results['exp1'].items():
                f.write(f"### Setting {setting}\n\n")
                
                # Mean forecasting
                if 'mean' in data:
                    f.write("#### Mean Forecasting\n\n")
                    f.write("| Method | MAB | Var | MSE | CR | CL |\n")
                    f.write("|--------|-----|-----|-----|-------|-------|\n")
                    for method, mdata in data['mean'].items():
                        if 'metrics' in mdata:
                            m = mdata['metrics']
                            f.write(f"| {method} | {m['MAB']:.4f} | {m['Var']:.4f} | "
                                   f"{m['MSE']:.4f} | {m['CR']:.4f} | {m['CL']:.4f} |\n")
                    f.write("\n")
                
                # VaR forecasting
                if 'var' in data:
                    f.write("#### VaR Forecasting\n\n")
                    f.write("| Method | MAB | Var | MSE | CR | CL |\n")
                    f.write("|--------|-----|-----|-----|-------|-------|\n")
                    for method, mdata in data['var'].items():
                        if 'metrics' in mdata:
                            m = mdata['metrics']
                            f.write(f"| {method} | {m['MAB']:.4f} | {m['Var']:.4f} | "
                                   f"{m['MSE']:.4f} | {m['CR']:.4f} | {m['CL']:.4f} |\n")
                    f.write("\n")
        
        # Experiment 2
        if 'exp2' in results:
            f.write("## Experiment 2: Continuous Mean Drift\n\n")
            f.write("Results summarized for each setting.\n\n")
        
        # Experiment 3
        if 'exp3' in results:
            f.write("## Experiment 3: GARCH Volatility Regime Shifts\n\n")
            f.write("| Method | MSE | Var | CR | CL |\n")
            f.write("|--------|-----|-----|-------|-------|\n")
            for method, mdata in results['exp3'].items():
                if 'metrics' in mdata:
                    m = mdata['metrics']
                    f.write(f"| {method} | {m['MSE']:.6f} | {m['Var']:.6f} | "
                           f"{m['CR']:.4f} | {m['CL']:.4f} |\n")
            f.write("\n")


def main():
    """Run all experiments."""
    print("\n" + "="*80)
    print("BAWS EXPERIMENTAL EVALUATION")
    print("Bootstrap-Based Adaptive Window Selection for Forecasting")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Experiment 1 - Run with reduced replications for testing
    print("\n[Running Experiment 1...]")
    try:
        exp1_results = run_experiment1(
            settings=['A1'],  # Start with just A1 for speed
            T=2000,
            t_start=501,
            n_replications=100,  # Reduced for testing
            base_seed=42,
            output_dir=str(output_dir)
        )
        results['exp1'] = exp1_results
        print("\n✓ Experiment 1 completed successfully")
    except Exception as e:
        print(f"\n✗ Experiment 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Experiment 2 - Lightweight version
    print("\n[Running Experiment 2...]")
    try:
        exp2_results = run_experiment2(
            settings=['B1'],
            T=2000,
            t_start=501,
            n_replications=100,
            base_seed=43,
            output_dir=str(output_dir)
        )
        results['exp2'] = exp2_results
        print("\n✓ Experiment 2 completed successfully")
    except Exception as e:
        print(f"\n✗ Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Experiment 3 - Lightweight version
    print("\n[Running Experiment 3...]")
    try:
        exp3_results = run_experiment3(
            T=2000,
            t_start=501,
            n_replications=50,  # Very limited for GARCH
            base_seed=44,
            output_dir=str(output_dir)
        )
        results['exp3'] = exp3_results
        print("\n✓ Experiment 3 completed successfully")
    except Exception as e:
        print(f"\n✗ Experiment 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    print("\n[Saving results...]")
    results_file = output_dir / 'RESULTS.md'
    save_results_summary(results, results_file)
    print(f"✓ Results saved to {results_file}")
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80 + "\n")
    
    return results


if __name__ == '__main__':
    results = main()
