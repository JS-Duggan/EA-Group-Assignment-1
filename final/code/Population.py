from Individual import Individual
from permutation import Permutation
from tsp import TSP

class Population:
    """
    Stores the information about a population of Individuals
    """

    def __init__(self, pop_size, tsp: TSP):
        """
        Generates a set of pop_size individuals to act as the generation
        Args:
            pop_size (int): the size of the population in each generation
            tsp (TSP): the TSP instance that the algorithm is trying to solve
        """

        self.individuals = []
        for _ in range(pop_size):
            gen_ind = Individual(tsp)
            self.individuals.append(gen_ind)
    
    def get_population(self):
        """
        Returns a list of all permutations present in the population
        Outputs:
            all_paths (list[list[int]]): all permutations (paths) represented in the popultation
        """

        all_paths = []
        for gen_ind in self.individuals:
            all_paths.append(gen_ind.get_path())
        return all_paths
    
    def get_population_cost(self):
        """
        Returns a list of all the costs present in the population
        Returns:
            individual_costs (list[int]): all costs represented in the popultation
        """
        individual_costs = []
        for ind in self.individuals:
            cost = ind.get_cost()
            individual_costs.append(cost)
        return individual_costs
            
