#!/usr/bin/env python3
"""
Long-run comparison of the three evolutionary algorithms.
Tests each algorithm with population size 20 on a specified TSP problem for:
2000, 5000, 10000, and 20000 generations
"""

import time
import csv
import os
import random
import argparse
import gc
from evolution import EvolutionaryAlgorithm
from tsp import TSP

def initialize_csv_file(filename):
    """Initialize CSV file with headers."""
    os.makedirs("./saves", exist_ok=True)
    filepath = os.path.join("./saves", filename)
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Generations', 'Best_Cost', 'Avg_Cost', 'Worst_Cost', 
                     'Runtime_Seconds', 'Improvement', 'Improvement_Percent', 'Population_Diversity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return filepath

def append_results_to_csv(results, filepath):
    """Append results to existing CSV file."""
    with open(filepath, 'a', newline='') as csvfile:
        fieldnames = ['Algorithm', 'Generations', 'Best_Cost', 'Avg_Cost', 'Worst_Cost', 
                     'Runtime_Seconds', 'Improvement', 'Improvement_Percent', 'Population_Diversity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for result in results:
            writer.writerow(result)
    
    print(f"Results appended to: {filepath}")

def calculate_stats(population, tsp_instance):
    """Calculate population statistics."""
    costs = [tsp_instance.permutationCost(individual) for individual in population]
    return {
        'best_cost': min(costs),
        'worst_cost': max(costs),
        'avg_cost': sum(costs) / len(costs)
    }

def run_algorithm_test_continuous(algorithm_name, algorithm_func, tsp_instance, initial_population, 
                                 checkpoint_generations, initial_best_cost, csv_filepath):
    """Run an algorithm continuously for max generations, saving at checkpoints."""
    print(f"\n{'='*80}")
    print(f"TESTING {algorithm_name.upper()} ALGORITHM")
    print(f"{'='*80}")
    
    max_generations = max(checkpoint_generations)
    print(f"Running {algorithm_name} for {max_generations:,} generations with checkpoints at: {checkpoint_generations}")
    
    # Create fresh copy of population for this test
    test_population = [ind.copy() for ind in initial_population]
    
    # Create EA instance for this algorithm run
    ea = EvolutionaryAlgorithm(tsp_instance, len(test_population))
    ea.tsp = tsp_instance
    ea.population = [ind.copy() for ind in test_population]
    ea.population_size = len(test_population)
    ea.evaluate_population()
    
    start_time = time.time()
    results = []
    
    # Determine algorithm configuration
    if algorithm_name == 'BALANCED':
        config = {
            'selection_method': 'tournament',
            'crossover_method': 'pmx', 
            'mutation_method': 'swap',
            'mutation_rate': 0.1,
            'elitism_count': 2
        }
        print("Configuration: Tournament selection + PMX crossover + Swap mutation")
    elif algorithm_name == 'EXPLORATION':
        config = {
            'selection_method': 'roulette',
            'crossover_method': 'cycle',
            'mutation_method': 'insert', 
            'mutation_rate': 0.2,
            'elitism_count': 1
        }
        print("Configuration: Fitness-proportional selection + Cycle crossover + Insert mutation")
    elif algorithm_name == 'EXPLOITATION':
        config = {
            'selection_method': 'tournament',
            'crossover_method': 'order',
            'mutation_method': 'swap',
            'mutation_rate': 0.05,
            'elitism_count': 4
        }
        print("Configuration: Tournament selection + Order crossover + Swap mutation")
    
    # Run generations continuously
    for generation in range(max_generations):
        # Evolve one generation
        ea.evolve_generation(**config)
        
        current_generation = generation + 1
        
        # Check if this is a checkpoint generation
        if current_generation in checkpoint_generations:
            checkpoint_time = time.time()
            runtime = checkpoint_time - start_time
            
            # Calculate current statistics
            current_stats = calculate_stats(ea.population, tsp_instance)
            best_cost = ea.best_fitness
            
            # Calculate improvement
            improvement = initial_best_cost - best_cost
            improvement_percent = (improvement / initial_best_cost) * 100
            
            # Calculate diversity
            diversity = ea.get_population_diversity()
            
            result = {
                'Algorithm': algorithm_name,
                'Generations': current_generation,
                'Best_Cost': round(best_cost, 2),
                'Avg_Cost': round(current_stats['avg_cost'], 2),
                'Worst_Cost': round(current_stats['worst_cost'], 2),
                'Runtime_Seconds': round(runtime, 2),
                'Improvement': round(improvement, 2),
                'Improvement_Percent': round(improvement_percent, 1),
                'Population_Diversity': round(diversity, 2)
            }
            
            results.append(result)
            
            print(f"  Checkpoint at {current_generation:,} generations:")
            print(f"    Best cost: {best_cost:.2f}")
            print(f"    Improvement: {improvement:.2f} ({improvement_percent:.1f}%)")
            print(f"    Diversity: {diversity:.2f}")
            print(f"    Runtime so far: {runtime:.1f} seconds")
        
        # Print progress every 5000 generations
        elif current_generation % 5000 == 0:
            current_best = min(ea.fitness_scores)
            current_avg = sum(ea.fitness_scores) / len(ea.fitness_scores)
            current_diversity = ea.get_population_diversity()
            print(f"Generation {current_generation:5d}: Best={current_best:8.2f}, Avg={current_avg:8.2f}, Diversity={current_diversity:.2f}")
    
    final_time = time.time()
    total_runtime = final_time - start_time
    
    print(f"{algorithm_name} completed in {total_runtime:.1f} seconds")
    print(f"Final best cost: {ea.best_fitness:.2f}")
    
    # Write all results for this algorithm to CSV immediately
    append_results_to_csv(results, csv_filepath)
    
    # Clean up memory
    del ea, test_population
    gc.collect()
    
    return results

def main():
    """Main function to run the comprehensive comparison."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Long-run comparison of evolutionary algorithms on TSP problems')
    parser.add_argument('test_case', nargs='?', default='kroA100', 
                       help='TSP test case name (without .tsp extension), e.g., kroA100, eil51, etc.')
    args = parser.parse_args()
    
    test_case = args.test_case
    
    print("Long-Run Evolutionary Algorithm Comparison")
    print("=" * 80)
    print("Population size: 20")
    print(f"Problem: {test_case}")
    print("Continuous run to 20,000 generations with checkpoints at: 2,000 | 5,000 | 10,000 | 20,000")
    
    # Initialize TSP problem
    tsp_file = f"test_cases/{test_case}.tsp"
    if not os.path.exists(tsp_file):
        print(f"Error: TSP file '{tsp_file}' not found!")
        print("Available test cases:")
        test_dir = "test_cases"
        if os.path.exists(test_dir):
            for file in sorted(os.listdir(test_dir)):
                if file.endswith('.tsp'):
                    print(f"  {file[:-4]}")  # Remove .tsp extension
        return
    
    tsp = TSP(tsp_file, "dummy_save.csv")
    print(f"\nTSP Problem: {tsp_file}")
    print(f"Problem dimension: {tsp.dimension} cities")
    
    # Set random seed for reproducible results
    random.seed(42)
    print("Random seed: 42")
    
    # Initialize CSV file for incremental writing
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"long_run_comparison_{test_case}_{timestamp}.csv"
    csv_filepath = initialize_csv_file(filename)
    print(f"Results will be saved to: {csv_filepath}")
    
    # Create evolutionary algorithm instance
    ea = EvolutionaryAlgorithm(tsp, population_size=20)
    
    # Create initial population (same for all tests)
    population_size = 20
    initial_population = []
    for _ in range(population_size):
        individual = tsp.generate_random_path(tsp.dimension)
        initial_population.append(individual)
    
    # Calculate initial statistics
    initial_stats = calculate_stats(initial_population, tsp)
    initial_best_cost = initial_stats['best_cost']
    
    print(f"\nInitial Population Statistics:")
    print(f"Best cost:    {initial_stats['best_cost']:8.2f}")
    print(f"Average cost: {initial_stats['avg_cost']:8.2f}")
    print(f"Worst cost:   {initial_stats['worst_cost']:8.2f}")
    
    # Checkpoint generations for continuous runs
    checkpoint_generations = [2000, 5000, 10000, 20000]
    
    # Store all results for final summary
    all_results = []
    
    # Test each algorithm with continuous runs
    algorithms = [
        ('BALANCED', None),
        ('EXPLORATION', None),
        ('EXPLOITATION', None)
    ]
    
    for algorithm_name, _ in algorithms:
        results = run_algorithm_test_continuous(
            algorithm_name, None, tsp, initial_population, 
            checkpoint_generations, initial_best_cost, csv_filepath
        )
        all_results.extend(results)
        
        # Clean up memory after each algorithm
        gc.collect()
    
    # Print summary table
    print(f"\n{'='*120}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*120}")
    print(f"{'Algorithm':<12} {'Generations':<12} {'Best Cost':<12} {'Improvement':<12} {'Runtime (s)':<12} {'Diversity':<10}")
    print("-" * 120)
    
    for result in all_results:
        print(f"{result['Algorithm']:<12} {result['Generations']:<12,} {result['Best_Cost']:<12.2f} "
              f"{result['Improvement']:<12.2f} {result['Runtime_Seconds']:<12.1f} {result['Population_Diversity']:<10.2f}")
    
    # Find best results for each generation count
    print(f"\n{'='*80}")
    print("BEST ALGORITHM FOR EACH GENERATION COUNT")
    print(f"{'='*80}")
    
    for generations in checkpoint_generations:
        gen_results = [r for r in all_results if r['Generations'] == generations]
        best_result = min(gen_results, key=lambda x: x['Best_Cost'])
        print(f"{generations:>6,} generations: {best_result['Algorithm']:<12} (Cost: {best_result['Best_Cost']:7.2f})")
    
    # Overall best result
    overall_best = min(all_results, key=lambda x: x['Best_Cost'])
    print(f"\nOVERALL BEST RESULT:")
    print(f"Algorithm: {overall_best['Algorithm']}")
    print(f"Generations: {overall_best['Generations']:,}")
    print(f"Best cost: {overall_best['Best_Cost']:.2f}")
    print(f"Improvement: {overall_best['Improvement']:.2f} ({overall_best['Improvement_Percent']:.1f}%)")
    print(f"Runtime: {overall_best['Runtime_Seconds']:.1f} seconds")
    
    # Clean up dummy file and final memory cleanup
    if os.path.exists("dummy_save.csv"):
        os.remove("dummy_save.csv")
    
    del all_results, ea, initial_population
    gc.collect()
    
    print(f"\nComparison completed! Results saved to: {csv_filepath}")
    print(f"Usage: python {os.path.basename(__file__)} [test_case_name]")
    print(f"Example: python {os.path.basename(__file__)} eil51")

if __name__ == "__main__":
    main()
