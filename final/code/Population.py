from individual import Individual
from tsp import TSP

class Population:
    """
    Stores the information about a population of Individuals
    """

    def __init__(self, pop_size, tsp: TSP):
        """
        Generates a set of pop_size individuals to act as the generation

        Inputs:
            pop_size (int): the size of the population in each generation
            tsp (TSP): the TSP instance that the algorithm is trying to solve
        """

        self.individuals = []
        for _ in range(pop_size):
            gen_ind = Individual(tsp)
            self.individuals.append(gen_ind)
    
    def getPopulation(self):
        """
        Returns a list of all permutations present in the population

        Outputs:
            all_paths (list[int]): all permutations (paths) represented in the popultation
        """

        all_paths = []
        for gen_ind in self.individuals:
            all_paths.append(gen_ind.getPath())
        return all_paths