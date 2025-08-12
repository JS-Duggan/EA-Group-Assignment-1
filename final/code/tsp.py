import random
import os
import csv
import typing

from load_tsp import loadTSP

class TSP:
    graph = [[]]
    saveFile: typing.TextIO
    csvWriter: csv.writer

    def __init__(self, testPath, savePath):
        """Init class

        takes path to test case as input
        edits private variable 'graph'
        graph is 2d array, where graph[i][j] = distance between i and j
        """
        
        loader = loadTSP(testPath)
        
        self.graph = loader.get_distance_matrix()    
        self.dimension = loader.get_dimension()
        
        self.loadSaveFile(savePath)
        return

    def getDimension(self):
        return self.dimension
    
    def permutationCost(self, permutation):
        cost = 0
        for i in range(len(permutation) - 1):
            cost += self.graph[permutation[i] * self.dimension + permutation[i + 1]]
        return cost
    
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
        # Handle invalid swap cases
        n = len(perm)

        if i == j or n < 2:
            return cost
        
        if i > j:
            i, j = j, i

        # Distance lookup helper
        def dist(x, y):
            return self.graph[x * n + y]

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
                sol[i], sol[j] = sol[j], sol[i]
                return sol, n_cost
        return sol, cost
    
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
        # Calculate changed cost entering inversion
        if i > 0:
            cost -= self.graph[perm[i - 1] * self.dimension + perm[i]]
            cost += self.graph[perm[i - 1] * self.dimension + perm[j]]
        
        # Calculate changed cost exiting inversion
        if j < len(perm) - 1:
            cost -= self.graph[perm[j] * self.dimension + perm[j + 1]]
            cost += self.graph[perm[i] * self.dimension + perm[j + 1]]
            
        return cost
    
    def inversion(self, perm, cost):
        """
        Perform the inversion neighbourhood search until a better solution is found
        if no better solution is found, returns input permutation and cost.
        Inversion range is random.
        Args:
            perm (list[int]): Current tour
            cost (float): Cost of the tour
        Returns:
            (list[int], float): The improved permutation and its cost.
        """
        pairs = self.random_pairs(self.dimension)
        
        for i, j in pairs:         
            new_cost = self.delta_inversion_cost(perm, cost, i, j)
            
            # Improved cost
            if new_cost < cost:
                # Perform inversion (inclusive of j as random pairs has j < n)
                new_perm = perm.copy()
                start = i
                end = j
                while start < end:
                    new_perm[start], new_perm[end] = new_perm[end], new_perm[start]
                    start += 1
                    end -= 1
                
                # Return permutation and cost of cheaper path
                return new_perm, new_cost
            
        # No cost improvement
        return perm, cost
    
    def jump(self, perm, cost):
        """
        Perform the jump neighbourhood search using delta evaluation.
        Moves a city from position i to j and shifts others accordingly.

        Args:
            perm (list[int]): Current tour
            cost (float): Cost of the tour

        Returns:
            (list[int], float): The improved permutation and its cost.
        """
        n = len(perm)
        indices = [(i, j) for i in range(n) for j in range(n) if i != j]
        random.shuffle(indices)

        for i, j in indices:
            # Create new tour by moving city from i to j
            new_perm = perm.copy()
            city = new_perm.pop(i)
            new_perm.insert(j, city)

            # Compute cost of new tour (delta evaluation)
            new_cost = 0
            for k in range(n):
                new_cost += self.graph[new_perm[k] * self.dimension + new_perm[(k + 1) % n]]

            if new_cost < cost:
                return new_perm, new_cost

        return perm, cost
    
    def localSearch(self, basePerm, nIterations):
        """
        Performs localSearch to determine an optimised route to the Traveling Salesman Problem 
        Results are saved to a csv file for processing later

        Args:
            basePerm (list[int]): Initial tour
            nIterations (striintng): The number of attempts the algorithm will have to produce an optimised value 

        Returns:
            """
        
        for i in range(nIterations):
            # Calculate the overall cost
            baseCost = self.permutationCost(basePerm)
            
            # Calculate results for the jump
            jumpCost = baseCost
            jumpPerm = basePerm
            while True:
                jumpPerm, tempCost = self.jump(jumpPerm, jumpCost)
                if (tempCost < jumpCost):
                    jumpCost = tempCost
                else:
                    break

            # Calculate results for the exchange
            exchCost = baseCost
            exchPerm = basePerm
            while True:
                exchPerm, tempCost = self.exchange(exchPerm, exchCost)
                if (tempCost < exchCost):
                    exchCost = tempCost
                else:
                    break

            # Calculate results for the inversion
            invsCost = baseCost
            invsPerm = basePerm
            while True:
                invsPerm, tempCost = self.inversion(invsPerm, invsCost)
                if (tempCost < invsCost):
                    invsCost = tempCost
                else:
                    break


            self.saveData(jumpPerm, jumpCost, exchPerm, exchCost, invsPerm, invsCost)


        
    def loadSaveFile(self, filePath):
        """
        Prepares the save file for the algorithm. It creates a new file if required, and loads it into memory. 
        If the file already exists, then a new file will be created with a slightly modified name (#)

        Args:
            filePath (string): file path to the csv file where the data will be saved. 

        Returns:
            """
        
        # if the file exists, give it a unique name so that it does not get overriden
        tempFileName = filePath
        pos = len(filePath) - 4 # ignore the .csv at the end
        n = 1
        while os.path.exists(tempFileName):
            tempFileName = filePath[:pos] + '(' + n.__str__() + ')' + filePath[pos:]
            n += 1
        filePath = tempFileName

        # Generate the file and the csv writer for use
        self.saveFile = open(filePath, 'w', newline='')
        self.csvWriter = csv.writer(self.saveFile)
        self.csvWriter.writerow(['Jump - tour', 'Jump - cost', 'Exchange - tour', 'Exchange - cost', 'Inverse - tour', 'Inverse - cost'])
        self.saveFile.flush()

    def saveData(self, jumpTour, jumpCost, exchangeTour, exchangeCost, inverseTour, inverseCost):
        """
        Saves an entry of data into the csv file specified in the constructor. 
        This entry has the result for the three different switching types. 

        Args:
            jumpTour (list[int]): the calculated best tour when using jump
            jumpCost (int): the calculated cos for the tour found using jump 
            exchangeTour (list[int]): the calculated best tour when using exchange
            exchangeCost (int): the calculated cos for the tour found using exchange 
            inverseTour (list[int]): the calculated best tour when using inverse
            inverseCost (int): the calculated cos for the tour found using inverse 

        Returns:
            """
        self.csvWriter.writerow([jumpTour, jumpCost, exchangeTour, exchangeCost, inverseTour, inverseCost])
        self.saveFile.flush()