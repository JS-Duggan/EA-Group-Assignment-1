import random
import os
import csv
import typing
import time
import sys
import numpy as np

from load_tsp import loadTSP
from permutation import Permutation

class TSP(Permutation):
    graph = [[]]
    saveFile: typing.TextIO
    csvWriter: csv.writer

    def generate_random_path(self, num_nodes):
        path = list(range(0, num_nodes))
        random.shuffle(path)
        return path

    def __init__(self, testPath, savePath):
        """Init class

        takes path to test case as input
        edits private variable 'graph'
        graph is 2d array, where graph[i][j] = distance between i and j
        """
        

        
        super().__init__(testPath)
        
        self.loadSaveFile(savePath)
        return
    
    
#     def permutationCost(self, perm):
#         n = len(perm)
#         cost = 0
#         for i in range(n):
#             cost += self.graph[perm[i], perm[(i + 1) % n]]
#         return cost
    
    # def random_pairs(self, n):
    #     # Generate all unique index pairs (i, j) where i < j
    #     pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    #     random.shuffle(pairs)  # Randomize the order of swaps
    #     return pairs
    
#     def random_pair(self, n):
#         """
#         Generate a single random pair i, j where i < j < n
        
#         Args:
#             n (int): Number of cities
        
#         Returns:
#             (int, int): Random pair (i, j)
#         """
#         i = random.randint(0, n - 2)
#         j = random.randint(i + 1, n - 1)
#         return i, j
        
#     def delta_swap_cost(self, perm, cost, i, j):
#         """
#         Calculate the new tour cost if cities at positions i and j are swapped,
#         using delta evaluation instead of recalculating the whole cost.

#         Args:
#             perm (list[int]): Current tour
#             cost (float): Cost of the tour
#             i (int): First swap position
#             j (int): Second swap position

#         Returns:
#             float: New tour cost after swapping i and j"""
#         # Handle invalid swap cases
#         n = len(perm)

#         if i == j or n < 2:
#             return cost
        
#         if i > j:
#             i, j = j, i

#         # Distance lookup helper
#         def dist(x, y):
#             return self.graph[x, y]

#         # Neighbors of i and j
#         a, b, c = perm[(i - 1) % n], perm[i], perm[(i + 1) % n]
#         d, e, f = perm[(j - 1) % n], perm[j], perm[(j + 1) % n]

#         # Handle adjacency and non-adjacency cases
#         if (j - i) == 1:
#             old_cost = dist(a, b) + dist(b, e) + dist(e, f)
#             new_cost = dist(a, e) + dist(e, b) + dist(b, f)
#         elif i == 0 and j == n - 1:
#             old_cost = dist(d, e) + dist(e, b) + dist(b, c)
#             new_cost = dist(d, b) + dist(b, e) + dist(e, c)
#         else:
#             old_cost = dist(a, b) + dist(b, c) + dist(d, e) + dist(e, f)
#             new_cost = dist(a, e) + dist(e, c) + dist(d, b) + dist(b, f)

#         # Return updated cost
#         return cost + (new_cost - old_cost)


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
        n = len(perm)

        # pairs = self.random_pairs(n)
        
        max_pairs = n * (n - 1) // 2 # Number of possible pairs where i < j
        for _ in range(max_pairs):
            i, j = self.random_pair(n)
            n_cost = self.delta_swap_cost(perm, cost, i, j)
            if n_cost < cost - 1e-9:

#                 sol[i], sol[j] = sol[j], sol[i]    
                
#                 return sol, n_cost
#         return sol, cost
    
#     def delta_inversion_cost(self, perm, cost, i, j):
#         """
#         Calculate the cost after inversion between i and j.
#         Only the costs entering (i - 1 to i) and exiting (j to j + 1) the inversion change
#         as the graph is undirected.
#         Args:
#             perm (list[int]): Current tour
#             cost (float): Cost of the tour
#             i (int): Inversion start
#             j (int): Inversion end

#         Returns:
#             float: New tour cost after inversion between i and j
#         """
#         n = len(perm)
        
#         if i == 0 and j == n-1:
#             return cost
        
#         # Remove edges
#         cost -= self.graph[perm[(i - 1) % n], perm[i]]
#         cost -= self.graph[perm[j], perm[(j + 1) % n]]
        
#         # Add edges
#         cost += self.graph[perm[(i - 1) % n], perm[j]]
#         cost += self.graph[perm[i], perm[(j + 1) % n]]
            
#         return cost

                return self.swap_pair(perm.copy(), i, j), n_cost
        return perm, cost

    
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
        n = len(perm)
        # pairs = self.random_pairs(n)
        
        max_pairs = n * (n - 1) // 2 # Number of possible pairs where i < j
        
        for _ in range(max_pairs):
            i, j = self.random_pair(n)
            new_cost = self.delta_inversion_cost(perm, cost, i, j)
            
            # Improved cost
            if new_cost < cost - 1e-9:

                # Perform inversion (inclusive of j as random pairs has j < n)
#                 new_perm = perm.copy()
#                 start = i
#                 end = j
#                 while start < end:
#                     new_perm[start], new_perm[end] = new_perm[end], new_perm[start]
#                     start += 1
#                     end -= 1
                            

                # Return permutation and cost of cheaper path
                return self.inversion_pair(perm, i, j), new_cost
            
        # No cost improvement
        return perm, cost
    

#     def delta_jump_cost(self, perm, cost, i, j):
#         """
#         Calculate the cost after jump i to j, using delta evaluation instead of
#         recalculating the whole cost.

#         Args:
#             perm (list[int]): Current tour
#             cost (float): Cost of the tour
#             i (int): Jump from
#             j (int): Jump to

#         Returns:
#             float: New tour cost after jump
#         """
#         n = len(perm)
        
#         if (i == 0 and j == n-1) or (i == n-1 and j == 0): 
#             return cost
        
#         if i < j:
#             # Remove edges
#             cost -= self.graph[perm[(i - 1) % n], perm[i]]
#             cost -= self.graph[perm[i], perm[(i + 1) % n]]
#             cost -= self.graph[perm[j], perm[(j + 1) % n]]
            
#             # Add edges
#             cost += self.graph[perm[(i - 1) % n], perm[(i + 1) % n]]
#             cost += self.graph[perm[i], perm[(j + 1) % n]]
#             cost += self.graph[perm[j], perm[i]]
#         else:
#             # Remove edges
#             cost -= self.graph[perm[(i - 1) % n], perm[i]]
#             cost -= self.graph[perm[i], perm[(i + 1) % n]]
#             cost -= self.graph[perm[(j - 1) % n], perm[j]]
            
#             # Add edges
#             cost += self.graph[perm[(i - 1) % n], perm[(i + 1) % n]]
#             cost += self.graph[perm[(j - 1) % n], perm[i]]
#             cost += self.graph[perm[i], perm[j]]
        
#         return cost
    

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
        # indices = [(i, j) for i in range(n) for j in range(n) if i != j]
        
        # indices = [(i, j) for i in range(n) for j in range(n) if i < j]
        # random.shuffle(indices)

        max_pairs = n * (n - 1) // 2 # Number of possible pairs where i < j
        for _ in range(max_pairs):
            i, j = self.random_pair(n)
            
            # Jump(i, j)
            # Compute cost of new tour (delta evaluation)
            new_cost = self.delta_jump_cost(perm, cost, i, j)
                
            if new_cost < cost - 1e-9:
                # Create new tour by moving city from i to j
                return self.jump_pair(perm.copy(), i, j), new_cost
            
            # Reversed: Jump(j, i)
            # Compute cost of new tour (delta evaluation)
            new_cost = self.delta_jump_cost(perm, cost, j, i)
                
            if new_cost < cost - 1e-9:
                # Create new tour by moving city from i to j
                return self.jump_pair(perm.copy(), j, i), new_cost

        return perm, cost
    
    def localSearch(self, nIterations):
        """
        Performs localSearch to determine an optimised route to the Traveling Salesman Problem 
        Results are saved to a csv file for processing later

        Args:
            basePerm (list[int]): Initial tour
            nIterations (striintng): The number of attempts the algorithm will have to produce an optimised value 

        Returns:
            """
        
        iteration_limit = 170_000
        
        for i in range(nIterations):
            print(f"{i}:")
            
            # Generate random initial permutation
            basePerm = self.generate_random_path(self.dimension)
            
            # Calculate the overall cost
            baseCost = self.permutationCost(basePerm)
            
            checkpoint = time.perf_counter()
            
            # Calculate results for the jump
            print("Jump: ", end="")
            jumpCost = baseCost
            jumpPerm = basePerm
            i = 0
            while True:
                if i >= iteration_limit:
                    break
                
                jumpPerm, tempCost = self.jump(jumpPerm, jumpCost)
                if (tempCost < jumpCost):
                    jumpCost = tempCost
                else:
                    break
                    
                i += 1
                    
            
            # Output time taken
            print(f"{time.perf_counter() - checkpoint:.2f} seconds")
            checkpoint = time.perf_counter()


            # Calculate results for the exchange
            print("Exchange: ", end="")
            exchCost = baseCost
            exchPerm = basePerm
            i = 0
            while True:
                if i >= iteration_limit:
                    break
                
                exchPerm, tempCost = self.exchange(exchPerm, exchCost)
                if (tempCost < exchCost):                    
                    exchCost = tempCost
                else:
                    break
                
                i += 1
                    
                        
            # Output time taken
            print(f"{time.perf_counter() - checkpoint:.2f} seconds")
            checkpoint = time.perf_counter()
            
            # Calculate results for the inversion
            print("Inversion: ", end="")
            invsCost = baseCost
            invsPerm = basePerm
            i = 0
            while True:
                if i >= iteration_limit:
                    break
                
                invsPerm, tempCost = self.inversion(invsPerm, invsCost)
                if (tempCost < invsCost):                    
                    invsCost = tempCost
                else:
                    break
                
                i += 1
                

            # Output time taken
            print(f"{time.perf_counter() - checkpoint:.2f} seconds")
            print()


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
        
    def getSavePath(self):
        return self.saveFile.name