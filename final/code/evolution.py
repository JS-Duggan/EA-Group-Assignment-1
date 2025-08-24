
#!/usr/bin/env python3
"""
Evolutionary Algorithm for the Traveling Salesman Problem (TSP)

This script implements a genetic algorithm with:
- Population size: 50 individuals
- Configurable number of generations via command line argument
- Tournament and roulette wheel selection operators
- Order crossover, partially mapped crossover, and cycle crossover
- Mutation operators using swap, inversion, and jump operations
- Elitism preservation strategy

Usage:
    python evolution.py <generations> <tsp_file> [save_path]
"""

import argparse
import random
import time
import csv
import os
import sys
from typing import List, Tuple, Optional

from tsp import TSP
from crossover import Crossover
from Population import Population
from Individual import Individual


class EvolutionaryAlgorithm:
    """Evolutionary Algorithm implementation for TSP optimization."""
    
    def __init__(self, tsp_instance: TSP, population_size: int = 50):
        """
        Initialize the evolutionary algorithm.
        
        Args:
            tsp_instance: TSP problem instance
            population_size: Size of the population (default: 50)
        """
        self.tsp = tsp_instance
        self.population_size = population_size
        self.crossover = Crossover()
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = float('inf')
        self.generation_stats = []
        
        # Selection operators
        self.selection_operators = {
            'tournament': self.tournament_selection,
            'roulette': self.roulette_wheel_selection
        }
        
        # Crossover operators
        self.crossover_operators = {
            'order': self.crossover.order_crossover,
            'pmx': self.crossover.partially_mapped_crossover,
            'cycle': self.crossover.cycle_crossover
        }
        
        # Mutation operators
        self.mutation_operators = {
            'swap': self._mutation_swap,
            'inversion': self._mutation_inversion,
            'jump': self._mutation_jump
        }

    def initialize_population(self) -> None:
        """Initialize the population with random individuals."""
        self.population = []
        for _ in range(self.population_size):
            individual = self.tsp.generate_random_path(self.tsp.dimension)
            self.population.append(individual)
        
        # Evaluate initial population
        self.evaluate_population()
        print(f"Population initialized with {self.population_size} individuals")

    def evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in the population."""
        self.fitness_scores = []
        for individual in self.population:
            fitness = self.tsp.permutationCost(individual)
            self.fitness_scores.append(fitness)
            
            # Track best individual
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual.copy()

    def tournament_selection(self, tournament_size: int = 3) -> List[int]:
        """
        Tournament selection operator.
        
        Args:
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual (tour)
        """
        tournament_indices = random.sample(range(self.population_size), tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return self.population[winner_idx].copy()

    def roulette_wheel_selection(self) -> List[int]:
        """
        Roulette wheel selection operator (fitness proportionate selection).
        For minimization problems, we use inverse fitness.
        
        Returns:
            Selected individual (tour)
        """
        # Convert to maximization problem by using inverse fitness
        # Add small constant to avoid division by zero
        inverse_fitness = [1.0 / (f + 1.0) for f in self.fitness_scores]
        total_fitness = sum(inverse_fitness)
        
        # Roulette wheel spin
        spin = random.uniform(0, total_fitness)
        current_sum = 0
        
        for i, fitness in enumerate(inverse_fitness):
            current_sum += fitness
            if current_sum >= spin:
                return self.population[i].copy()
        
        # Fallback (should not reach here)
        return self.population[-1].copy()

    def _mutation_swap(self, individual: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Swap mutation operator.
        
        Args:
            individual: Tour to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() < mutation_rate:
            n = len(individual)
            i, j = self.tsp.random_pair(n)
            individual = self.tsp.swap_pair(individual.copy(), i, j)
        return individual

    def _mutation_inversion(self, individual: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Inversion mutation operator.
        
        Args:
            individual: Tour to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() < mutation_rate:
            n = len(individual)
            i, j = self.tsp.random_pair(n)
            individual = self.tsp.inversion_pair(individual, i, j)
        return individual

    def _mutation_jump(self, individual: List[int], mutation_rate: float = 0.1) -> List[int]:
        """
        Jump mutation operator.
        
        Args:
            individual: Tour to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() < mutation_rate:
            n = len(individual)
            i, j = self.tsp.random_pair(n)
            individual = self.tsp.jump_pair(individual.copy(), i, j)
        return individual

    def _validate_individual(self, individual: List[int], fallback_parent: List[int]) -> List[int]:
        """
        Validate and fix an individual by filling any None values.
        
        Args:
            individual: Individual to validate
            fallback_parent: Parent to use for missing values
            
        Returns:
            Validated individual with no None values
        """
        if individual is None:
            return fallback_parent.copy()
        
        # Check for None values and replace them
        all_values = set(range(self.tsp.dimension))
        used_values = set(val for val in individual if val is not None)
        missing_values = list(all_values - used_values)
        
        validated_individual = individual.copy()
        missing_idx = 0
        
        for i, val in enumerate(validated_individual):
            if val is None:
                if missing_idx < len(missing_values):
                    validated_individual[i] = missing_values[missing_idx]
                    missing_idx += 1
                else:
                    # Fallback to parent value
                    validated_individual[i] = fallback_parent[i]
        
        return validated_individual

    def evolve_generation(self, selection_method: str = 'tournament', 
                         crossover_method: str = 'order',
                         mutation_method: str = 'swap',
                         mutation_rate: float = 0.1,
                         elitism_count: int = 2) -> None:
        """
        Evolve one generation of the population.
        
        Args:
            selection_method: Selection operator to use
            crossover_method: Crossover operator to use
            mutation_method: Mutation operator to use
            mutation_rate: Probability of mutation
            elitism_count: Number of best individuals to preserve
        """
        # Store elite individuals
        elite_indices = sorted(range(len(self.fitness_scores)), 
                              key=lambda i: self.fitness_scores[i])[:elitism_count]
        elites = [self.population[i].copy() for i in elite_indices]
        
        # Create new population
        new_population = []
        
        # Add elites to new population
        new_population.extend(elites)
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self.selection_operators[selection_method]()
            parent2 = self.selection_operators[selection_method]()
            
            # Crossover
            child1, child2 = self.crossover_operators[crossover_method](parent1, parent2)
            
            # Validate crossover results (fill any None values)
            child1 = self._validate_individual(child1, parent1)
            child2 = self._validate_individual(child2, parent2)
            
            # Mutation
            child1 = self.mutation_operators[mutation_method](child1, mutation_rate)
            child2 = self.mutation_operators[mutation_method](child2, mutation_rate)
            
            # Add children to new population
            new_population.extend([child1, child2])
        
        # Ensure population size is exact
        self.population = new_population[:self.population_size]
        
        # Evaluate new population
        self.evaluate_population()

    def run_evolution(self, generations: int, 
                     selection_method: str = 'tournament',
                     crossover_method: str = 'order', 
                     mutation_method: str = 'swap',
                     mutation_rate: float = 0.1,
                     elitism_count: int = 2,
                     verbose: bool = True) -> Tuple[List[int], float]:
        """
        Run the evolutionary algorithm for specified generations.
        
        Args:
            generations: Number of generations to evolve
            selection_method: Selection operator ('tournament' or 'roulette')
            crossover_method: Crossover operator ('order', 'pmx', or 'cycle')
            mutation_method: Mutation operator ('swap', 'inversion', or 'jump')
            mutation_rate: Probability of mutation
            elitism_count: Number of elite individuals to preserve
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (best_tour, best_fitness)
        """
        print(f"Starting evolution with {generations} generations...")
        print(f"Parameters: selection={selection_method}, crossover={crossover_method}, "
              f"mutation={mutation_method}, mutation_rate={mutation_rate}, elitism={elitism_count}")
        
        start_time = time.time()
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(generations):
            # Evolve one generation
            self.evolve_generation(selection_method, crossover_method, 
                                 mutation_method, mutation_rate, elitism_count)
            
            # Collect statistics
            current_best = min(self.fitness_scores)
            current_avg = sum(self.fitness_scores) / len(self.fitness_scores)
            current_worst = max(self.fitness_scores)
            
            self.generation_stats.append({
                'generation': generation + 1,
                'best_fitness': current_best,
                'avg_fitness': current_avg,
                'worst_fitness': current_worst
            })
            
            # Print progress
            if verbose and (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1:4d}: Best={current_best:8.2f}, "
                      f"Avg={current_avg:8.2f}, Worst={current_worst:8.2f}")
        
        end_time = time.time()
        
        print(f"\nEvolution completed in {end_time - start_time:.2f} seconds")
        print(f"Best fitness found: {self.best_fitness:.2f}")
        print(f"Best tour: {self.best_individual}")
        
        return self.best_individual, self.best_fitness

    def save_results(self, save_path: str) -> None:
        """
        Save evolution results to a CSV file.
        
        Args:
            save_path: Path to save the results
        """
        # Create unique filename if file exists
        base_path = save_path
        counter = 1
        while os.path.exists(save_path):
            name, ext = os.path.splitext(base_path)
            save_path = f"{name}({counter}){ext}"
            counter += 1
        
        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['generation', 'best_fitness', 'avg_fitness', 'worst_fitness']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for stats in self.generation_stats:
                writer.writerow(stats)
        
        print(f"Results saved to: {save_path}")

    def get_population_diversity(self) -> float:
        """
        Calculate population diversity as average pairwise distance.
        
        Returns:
            Average diversity score
        """
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Calculate Hamming distance between tours
                distance = sum(1 for a, b in zip(self.population[i], self.population[j]) if a != b)
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0


def main():
    """Main function to run the evolutionary algorithm."""
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm for TSP')
    parser.add_argument('generations', type=int, help='Number of generations to run')
    parser.add_argument('tsp_file', type=str, help='Path to TSP problem file')
    parser.add_argument('--save_path', type=str, default=None, 
                       help='Path to save results (optional)')
    parser.add_argument('--selection', type=str, choices=['tournament', 'roulette'], 
                       default='tournament', help='Selection method')
    parser.add_argument('--crossover', type=str, choices=['order', 'pmx', 'cycle'], 
                       default='order', help='Crossover method')
    parser.add_argument('--mutation', type=str, choices=['swap', 'inversion', 'jump'], 
                       default='swap', help='Mutation method')
    parser.add_argument('--mutation_rate', type=float, default=0.1, 
                       help='Mutation rate (0.0-1.0)')
    parser.add_argument('--elitism', type=int, default=2, 
                       help='Number of elite individuals to preserve')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Validate arguments
    if args.generations <= 0:
        print("Error: Number of generations must be positive")
        sys.exit(1)
    
    if not os.path.exists(args.tsp_file):
        print(f"Error: TSP file '{args.tsp_file}' not found")
        sys.exit(1)
    
    if not (0.0 <= args.mutation_rate <= 1.0):
        print("Error: Mutation rate must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Set default save path if not provided
    if args.save_path is None:
        tsp_filename = os.path.splitext(os.path.basename(args.tsp_file))[0]
        args.save_path = f"evolution_results_{tsp_filename}.csv"
    
    try:
        # Initialize TSP instance
        print(f"Loading TSP problem: {args.tsp_file}")
        tsp = TSP(args.tsp_file, "dummy_save.csv")  # TSP requires save path but we won't use it
        print(f"Problem dimension: {tsp.dimension} cities")
        
        # Initialize evolutionary algorithm
        ea = EvolutionaryAlgorithm(tsp, population_size=50)
        
        # Run evolution
        best_tour, best_fitness = ea.run_evolution(
            generations=args.generations,
            selection_method=args.selection,
            crossover_method=args.crossover,
            mutation_method=args.mutation,
            mutation_rate=args.mutation_rate,
            elitism_count=args.elitism,
            verbose=True
        )
        
        # Save results
        ea.save_results(args.save_path)
        
        # Print final statistics
        final_diversity = ea.get_population_diversity()
        print(f"\nFinal population diversity: {final_diversity:.2f}")
        
        # Clean up dummy file if it was created
        dummy_file = "dummy_save.csv"
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()