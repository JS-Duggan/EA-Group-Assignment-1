from permutation import Permutation
from crossover import Crossover
from tsp import TSP

class Individual(Permutation):
    crossover: Crossover
    def __init__(self, tsp: TSP):
        num_nodes = tsp.dimension
        self.path = tsp.generate_random_path(num_nodes)
        
    def get_path(self):
        return self.path
 
    
    
    

