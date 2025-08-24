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

        Args:
            tsp (TSP): an instance of the TSP for generating a random path
        """

        super().__init__(tsp.dimension)
        self.tsp = tsp
        self.path = tsp.generate_random_path(tsp.dimension)
        
    def get_path(self):
        """
        Returns the path of the individual

        Returns:
            path (list[int]): the permutation of the TSP representing this individual
        """

        return self.path
    
    def get_cost(self):
        """
        Returns the cost of the individual path
        
        Returns:
            cost (int): the cost of the permutation of the TSP representing this individual
        """

        return self.tsp.permutationCost(self.path)

    

