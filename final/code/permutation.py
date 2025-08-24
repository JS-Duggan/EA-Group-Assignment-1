import random

from load_tsp import LoadTSP

class Permutation:
    """
    An base class that contains the basic mutation operators for a permutation
    """

    graph = [[]]
    dimension: int

    def __init__(self, test_path):
        """
        Generates the graph of distancesses between each pair of nodes for the TSP instance

        Inputs:
            test_path (string): the relative file location of the TSP isntance

        """
        
        tsp = LoadTSP(test_path)
        
        self.graph = tsp.getDistanceMatrix()
        self.dimension = tsp.getDimension()
    
    def permutationCost(self, perm):
        """
        Calculates the cost of a permutation through the TSP

        Inputs:
            perm (list[int]): the permutation through the TSP 

        Outputs:
            cost (int): the calcualted cost of the permutation through the TSP

        """
        n = len(perm)
        cost = 0
        for i in range(n):
            cost += self.graph[perm[i], perm[(i + 1) % n]]
        return cost
    
    def randomPair(self, n):
        """
        Generate a single random pair i, j where i < j < n
        
        Inputs:
            n (int): Number of cities
        
        Outputs:
            (int, int): Random pair (i, j)
        """
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        return i, j
    
    def deltaSwapCost(self, perm, cost, i, j):
        """
        Calculate the new tour cost if cities at positions i and j are swapped,
        using delta evaluation instead of recalculating the whole cost.

        Inputs:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
            i (int): First swap position
            j (int): Second swap position

        Outputs:
            cost (float): New tour cost after swapping i and j"""
        # Handle invalid swap cases
        n = len(perm)

        if i == j or n < 2:
            return cost
        
        if i > j:
            i, j = j, i

        # Distance lookup helper
        def dist(x, y):
            return self.graph[x, y]

        # Neighbors of i and j
        a, b, c = perm[(i - 1) % n], perm[i], perm[(i + 1) % n]
        d, e, f = perm[(j - 1) % n], perm[j], perm[(j + 1) % n]

        # Handle adjacency and non-adjacency cases
        if (j - i) == 1:
            old_cost = dist(a, b) + dist(b, e) + dist(e, f)
            new_cost = dist(a, e) + dist(e, b) + dist(b, f)
        elif i == 0 and j == n - 1:
            old_cost = dist(d, e) + dist(e, b) + dist(b, c)
            new_cost = dist(d, b) + dist(b, e) + dist(e, c)
        else:
            old_cost = dist(a, b) + dist(b, c) + dist(d, e) + dist(e, f)
            new_cost = dist(a, e) + dist(e, c) + dist(d, b) + dist(b, f)

        # Return updated cost
        return cost + (new_cost - old_cost)

    def swapPair(self, perm, i, j):
        """
        Perform swap between i and j node in the permutation

        Inputs:
            perm (list[int]): Current tour
            i (int): First swap position
            j (int): Second swap position

        Outputs:
            perm (list[int]): The resultant permutation 
        """
        perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def deltaInversionCost(self, perm, cost, i, j):
        """
        Calculate the cost after inversion between i and j.
        Only the costs entering (i - 1 to i) and exiting (j to j + 1) the inversion change
        as the graph is undirected.

        Inputs:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
            i (int): Inversion start
            j (int): Inversion end

        Outputs:
            cost (float): New tour cost after inversion between i and j
        """
        n = len(perm)
        
        if i == 0 and j == n-1:
            return cost
        
        # Remove edges
        cost -= self.graph[perm[(i - 1) % n], perm[i]]
        cost -= self.graph[perm[j], perm[(j + 1) % n]]
        
        # Add edges
        cost += self.graph[perm[(i - 1) % n], perm[j]]
        cost += self.graph[perm[i], perm[(j + 1) % n]]
            
        return cost
    
    def inversionPair(self, perm, i, j):
        """
        Perform inversion of i and j.  

        Inputs:
            perm (list[int]): Current tou
            i (int): Inversion start (must be less than j)
            j (int): Inversion end

        Outputs:
            new_perm (list[int]): The resultant permutation
        """
        
        new_perm = perm.copy()
        start = i
        end = j
        while start < end:
            new_perm[start], new_perm[end] = new_perm[end], new_perm[start]
            start += 1
            end -= 1
        return new_perm
    
    def deltaJumpCost(self, perm, cost, i, j):
        """
        Calculate the cost after jump i to j, using delta evaluation instead of
        recalculating the whole cost.

        Inputs:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
            i (int): Jump from
            j (int): Jump to

        Outputs:
            cost (float): New tour cost after jump
        """
        n = len(perm)
        
        if (i == 0 and j == n-1) or (i == n-1 and j == 0): 
            return cost
        
        if i < j:
            # Remove edges
            cost -= self.graph[perm[(i - 1) % n], perm[i]]
            cost -= self.graph[perm[i], perm[(i + 1) % n]]
            cost -= self.graph[perm[j], perm[(j + 1) % n]]
            
            # Add edges
            cost += self.graph[perm[(i - 1) % n], perm[(i + 1) % n]]
            cost += self.graph[perm[i], perm[(j + 1) % n]]
            cost += self.graph[perm[j], perm[i]]
        else:
            # Remove edges
            cost -= self.graph[perm[(i - 1) % n], perm[i]]
            cost -= self.graph[perm[i], perm[(i + 1) % n]]
            cost -= self.graph[perm[(j - 1) % n], perm[j]]
            
            # Add edges
            cost += self.graph[perm[(i - 1) % n], perm[(i + 1) % n]]
            cost += self.graph[perm[(j - 1) % n], perm[i]]
            cost += self.graph[perm[i], perm[j]]
            
        return cost
    
    def jumpPair(self, perm, i, j):
        """
        Perform jump on i to j

        Inputs:
            perm (list[int]): Current tour
            i (int): Jump from
            j (int): Jump to

        Outputs:
            perm (list[int]): The resutant permutation
        """
        city = perm.pop(i)
        perm.insert(j, city) 
        return perm