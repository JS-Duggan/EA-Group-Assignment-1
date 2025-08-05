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

    def cost(self, perm):
        """Cost
        
        given a permutation, uses graph to calculate total cost
        """
        cost = 0
        for i in range(len(perm) - 1):
            cost += self.graph[perm[i]][perm[i + 1]]
        cost += self.graph[perm[-1]][perm[0]]
        return cost

    def swap(self, perm, cost):
        """Swap
        
        uses class variable graph, along with an initial permutation/cost to 
        test for a better cost using the swap neighbourhood search.
        
        returns first permutation that has smaller cost
        """
        # create a copy of perm, not a reference
        sol = perm[:]
        for runs in range(30):
            new_found = False
            for i in range(len(sol) - 1):
                for j in range(i + 1, len(sol)):
                    sol[i], sol[j] = sol[j], sol[i]
                    n_cost = self.cost(sol)
                    if n_cost < cost:
                        new_found = True
                        cost = n_cost
                        break
                    sol[i], sol[j] = sol[j], sol[i]
                if new_found:
                    break
        return sol, cost