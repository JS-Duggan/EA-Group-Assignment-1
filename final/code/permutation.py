import random

from load_tsp import loadTSP

class Permutation:
    graph = [[]]
    dimension: int

    def __init__(self, testPath):
        """Init class

        takes path to test case as input
        edits private variable 'graph'
        graph is 2d array, where graph[i][j] = distance between i and j
        """
        
        tsp = loadTSP(testPath)
        
        self.graph = tsp.get_distance_matrix()
        self.dimension = tsp.get_dimension()

        
        return
    
    # def random_pairs(self, n):
    #     # Generate all unique index pairs (i, j) where i < j
    #     pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    #     random.shuffle(pairs)  # Randomize the order of swaps
    #     return pairs
    
    def permutationCost(self, perm):
        n = len(perm)
        cost = 0
        for i in range(n):
            cost += self.graph[perm[i], perm[(i + 1) % n]]
        return cost
    
    def random_pair(self, n):
        """
        Generate a single random pair i, j where i < j < n
        
        Args:
            n (int): Number of cities
        
        Returns:
            (int, int): Random pair (i, j)
        """
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        return i, j
    
    def delta_swap_cost(self, perm, cost, i, j):
        """
        Calculate the new tour cost if cities at positions i and j are swapped,
        using delta evaluation instead of recalculating the whole cost.

        Args:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
            i (int): First swap position
            j (int): Second swap position

        Returns:
            float: New tour cost after swapping i and j"""
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

    def swap_pair(self, perm, i, j):
        """
        Perform swap between i and j node in the permutation

        Args:
            perm (list[int]): Current tour
            i (int): First swap position
            j (int): Second swap position
        Returns:
            list[int]: The resultant permutation 
        """
        perm[i], perm[j] = perm[j], perm[i]
        return perm
    
    def delta_inversion_cost(self, perm, cost, i, j):
        """
        Calculate the cost after inversion between i and j.
        Only the costs entering (i - 1 to i) and exiting (j to j + 1) the inversion change
        as the graph is undirected.
        Args:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
            i (int): Inversion start
            j (int): Inversion end

        Returns:
            float: New tour cost after inversion between i and j
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
    
    def inversion_pair(self, perm, i, j):
        """
        Perform inversion of i and j.  

        Args:
            perm (list[int]): Current tou
            i (int): Inversion start (must be less than j)
            j (int): Inversion end
        Returns:
            list[int] : The resultant permutation
        """
        
        new_perm = perm.copy()
        start = i
        end = j
        while start < end:
            new_perm[start], new_perm[end] = new_perm[end], new_perm[start]
            start += 1
            end -= 1
        return new_perm
    
    def delta_jump_cost(self, perm, cost, i, j):
        """
        Calculate the cost after jump i to j, using delta evaluation instead of
        recalculating the whole cost.

        Args:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
            i (int): Jump from
            j (int): Jump to

        Returns:
            float: New tour cost after jump
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
    
    def jump_pair(self, perm, i, j):
        """
        Perform jump on i to j

        Args:
            perm (list[int]): Current tour
            i (int): Jump from
            j (int): Jump to

        Returns:
            list[int]: The resutant permutation
        """
        city = perm.pop(i)
        perm.insert(j, city) 
        return perm