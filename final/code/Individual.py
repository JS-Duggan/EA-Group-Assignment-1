<<<<<<< HEAD
from tsp import TSP

class Individual:
    def generate_single_permutation(self, num_nodes):
        individual_perm = TSP()
        individual_perm.generate_random_path(self, num_nodes)
        return individual_perm
    
=======
import random
from permutation import Permutation
from crossover import Crossover

class Individual(Permutation):
    crossover: Crossover
>>>>>>> origin/main

    def __init__(self):
        return
    
    
