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
        n = len(perm)

        # Helper function to get cost from 1D array
        def dist(x, y):
            return self.graph[x * n + y]

        # Wrap-around for circular tour
        a, b = perm[(i - 1) % n], perm[i]
        c = perm[(i + 1) % n]
        d, e = perm[(j - 1) % n], perm[j]
        f = perm[(j + 1) % n]

        # Adjacent swap case
        if j == i + 1 or (i == 0 and j == n - 1):
            old_cost = dist(a, b) + dist(e, f)
            new_cost = dist(a, e) + dist(b, f)
        # Non-adjacent swap case
        else:
            old_cost = dist(a, b) + dist(b, c) + dist(d, e) + dist(e, f)
            new_cost = dist(a, e) + dist(e, c) + dist(d, b) + dist(b, f)

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
    
    def inversion_cost(self, perm, i, j):
        """
        Calculate the cost of the path between i-1 and j+1 of the permutation.
        Inversion between i and j, will only change the cost between i-1 and j+1.
        
        Args:
            perm (list[int]): Current tour
            i (int): Inversion start
            j (int): Inversion end

        Returns:
            float: New cost between range i-1 and j+1 of permutation
        """
        
        cost = 0
        for source in range(i - 1, j + 1):
            if source < 0 or source == len(perm) - 1:
                continue
            
            dest = source + 1
            cost += self.graph[perm[source] * self.dimension + perm[dest]]
        
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
        
        new_perm = perm.copy()
        
        pairs = self.random_pairs(len(new_perm))
        
        for i, j in pairs:          
            new_cost = cost
            # Subtract old cost between i-1 and j+1 before inversion
            new_cost -= self.inversion_cost(new_perm, i, j)
            
            # Perform inversion (inclusive of j as random pairs has j < n)
            start = i
            end = j
            while start < end:
                new_perm[start], new_perm[end] = new_perm[end], new_perm[start]
                start += 1
                end -= 1
            
            # Add in the replaced cost between i-1 and j+1 after inversion
            new_cost += self.inversion_cost(new_perm, i, j)

            # Improved cost
            if new_cost < cost:
                return new_perm, new_cost
            
            cost = new_cost
        
        # No cost improvement
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
            '''while True:
                jumpPerm, tempCost = self.jump(jumpPerm, jumpCost)
                if (tempCost < jumpCost):
                    jumpCost = tempCost
                else:
                    break'''

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