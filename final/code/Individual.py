from permutation import Permutation
from crossover import Crossover
from tsp import TSP

class Individual(Permutation):
    crossover: Crossover
    def __init__(self, tsp: TSP):
        super().__init__(tsp.dimension)
        self.tsp = tsp
        self.path = tsp.generate_random_path(tsp.dimension)
        
    def get_path(self):
        return self.path
    
    def get_cost(self):
        return self.tsp.permutationCost(self.path)

    

