#!/usr/bin/env python3
"""
Demonstration script for the Evolutionary Algorithm

This script shows different ways to run the evolutionary algorithm
with various parameter combinations.
"""

import subprocess
import sys
import os

def run_evolution(generations, tsp_file, **kwargs):
    """
    Run the evolution script with given parameters.

    Args:
        generations (list[list[int]]): generation permutation list
        tsp_file (str): string name of TSP instance
        **kwargs (): additional arguments from command line
    """
    cmd = [sys.executable, "evolution.py", str(generations), tsp_file]
    
    for key, value in kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    result = subprocess.run(cmd, capture_output=False)
    print("-" * 50)
    print()
    
    return result.returncode == 0

def main():
    """Run demonstration examples."""
    print("Evolutionary Algorithm Demonstration")
    print("=" * 50)
    
    # Check if test files exist
    test_files = ["test_cases/eil51.tsp", "test_cases/st70.tsp"]
    available_files = [f for f in test_files if os.path.exists(f)]
    
    if not available_files:
        print("Error: No test files found!")
        return
    
    test_file = available_files[0]
    print(f"Using test file: {test_file}")
    print()
    
    # Example 1: Basic run with default parameters
    print("Example 1: Basic run (10 generations, default parameters)")
    run_evolution(10, test_file, seed=42)
    
    # Example 2: Tournament selection with PMX crossover
    print("Example 2: Tournament selection with PMX crossover")
    run_evolution(15, test_file, selection="tournament", crossover="pmx", 
                 mutation="swap", mutation_rate=0.15, seed=123)
    
    # Example 3: Roulette wheel selection with cycle crossover
    print("Example 3: Roulette wheel selection with cycle crossover")
    run_evolution(12, test_file, selection="roulette", crossover="cycle", 
                 mutation="inversion", mutation_rate=0.2, elitism=3, seed=456)
    
    # Example 4: High mutation rate with jump mutation
    print("Example 4: High mutation rate with jump mutation")
    run_evolution(8, test_file, selection="tournament", crossover="order", 
                 mutation="jump", mutation_rate=0.3, elitism=1, seed=789)
    
    print("Demonstration completed!")
    print("Check the generated CSV files for detailed results.")

if __name__ == "__main__":
    main()
