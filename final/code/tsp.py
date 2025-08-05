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

    def swap(self, perm, cost):
        """
        Performs up to 30 iterations of the swap neighbourhood search.
        Stops early if no improvement is found in a given iteration.
        Args:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
        Returns:
            (list[int], float): The improved permutation and its cost.
        """
        # create a copy of perm, not a reference
        sol = perm.copy()
        for runs in range(30):
            new_found = False
            for i in range(len(sol) - 1):
                for j in range(i + 1, len(sol)):
                    n_cost = self.delta_swap_cost(sol, cost, i, j)
                    if n_cost < cost:
                        sol[i], sol[j] = sol[j], sol[i]
                        cost = n_cost
                        new_found = True
                        break
                if new_found:
                    break
            if not new_found:
                break
        return sol, cost