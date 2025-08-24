from permutation import Permutation
from crossover import Crossover
from tsp import TSP

class Individual(Permutation):
    """
    A class that represents a single permutation of the TSP

    Individuals inherit the mutation opperators from the Permutation class and have cross over algorithms avaiable through the Crossover object
    """

    crossover: Crossover

    def __init__(self, tsp: TSP):
        """
        Setup the initial randomly generated path

        Inputs:
            tsp (TSP): an instance of the TSP for generating a random path
        """
        
        self.path = tsp.generateRandomPath(tsp.dimension)
        
    def getPath(self):
        """
        Returns the path of the individual

        Outputs:
            path (list[int]): the permutation of the TSP representing this individual
        """
        return self.path
 
    
    
    

