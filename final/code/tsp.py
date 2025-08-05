import random

class TSP:
    graph = [[]]

    def __init__(self, path):
        """Init class

        takes path to test case as input
        edits private variable 'graph'
        graph is 2d array, where graph[i][j] = distance between i and j
        """
        self.graph = [[]]
        return

    def random_pairs(self, n):
         # Generate all unique index pairs (i, j) where i < j
        pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        random.shuffle(pairs)  # Randomize the order of swaps
        return pairs

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
        n = len(perm)

        # Wrap-around for circular tour
        a, b = perm[(i - 1) % n], perm[i]
        c = perm[(i + 1) % n]
        d, e = perm[(j - 1) % n], perm[j]
        f = perm[(j + 1) % n]

        # Adjacent swap case
        if j == i + 1 or (i == 0 and j == n - 1):
            old_cost = self.graph[a][b] + self.graph[e][f]
            new_cost = self.graph[a][e] + self.graph[b][f]
        else:  # Non-adjacent case
            old_cost = (
                self.graph[a][b] + self.graph[b][c] +
                self.graph[d][e] + self.graph[e][f]
            )
            new_cost = (
                self.graph[a][e] + self.graph[e][c] +
                self.graph[d][b] + self.graph[b][f]
            )

        return cost + (new_cost - old_cost)

    def exchange(self, perm, cost):
        """
        Perform the exchange neighbourhood search until a better solution is found
        if no better solution is found, returns input permutation and cost.
        Algorithm is greedy, so swap paris are done randomly
        Args:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
        Returns:
            (list[int], float): The improved permutation and its cost.
        """
        # create a copy of perm, not a reference
        sol = perm.copy()
        n = len(sol)

        pairs = self.random_pairs(n)

        for i, j in pairs:
            n_cost = self.delta_swap_cost(sol, cost, i, j)
            if n_cost < cost:
                return sol, cost
        return sol, cost
    
    def inversion(self, perm, cost):
        perm_ = perm.copy()
        
        pairs = self.random_pairs(len(perm_))
        
        for i, j in pairs:
            old_cost = 0
            for i in range(i, j - 1):
                old_cost += self.graph[perm_[i]][perm_[i + 1]]
            
            # Perform inversion (inclusive of j as random pairs has j < n)
            for k, l in zip(range(i, j), range(j, i, -1)):
                temp = perm_[k]
                perm_[k] = perm_[l]
                perm_[l] = perm_[k]
            
            cost = 0
            for i in range(i, j - 1):
                cost += self.graph[perm_[i]][perm_[i + 1]]
                
            if cost < old_cost:
                return perm_, cost
        
        return perm_, cost
