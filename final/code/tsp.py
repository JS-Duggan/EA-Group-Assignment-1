import random
import os
import csv
import typing
import time
import numpy as np
from multiprocessing import shared_memory

from load_tsp import LoadTSP
from permutation import Permutation

class TSP(Permutation):
    """
    The main TSP algorithm which runs a greedy search approach to a TSP instance. 
    This class contains all the information and functions required to manage the TSP search process
    """
    graph = [[]]
    save_file: typing.TextIO
    csv_writer: csv.writer

    def generate_random_path(self, num_nodes):
        """
        Generates a random path through the TSP 
        Args:
            num_nodes (int): The number of nodes that are in the TSP instance
        
        Returns:
            path (list[int]): A randomly generated permutation of the nodes which acts as the path
        """
        path = list(range(0, num_nodes))
        random.shuffle(path)
        return path

    def __init__(self, testPath, savePath, dimension = None, sharedMemory = False):
        """Init class

        takes path to test case as input
        edits private variable 'graph'
        graph is 2d array, where graph[i][j] = distance between i and j
        
        Args:
            test_path (string): the relative file path to the TSP test instance
            save_path (string): the relative file path to the location where the raw data will be saved
        """
        
        self.dimension = dimension
        
        super().__init__(testPath, sharedMemory)
        
        self.loadSaveFile(savePath)
        return
    
    def load_shared_memory(self, shmName: str, shape, dtype):
        """
        Loads the data for shared memory
        
        Args:
            shmName (str): shared memory name
            shape (Shape): shape of the data
            dtype (Data Type): data type of the data
        """
        
        # Set reference to graph in shared memory
        self._shm = shared_memory.SharedMemory(name=shmName)
        self.graph = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)

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
                # Return permutation and cost of cheaper path
                return self.inversion_pair(perm, i, j), new_cost
            
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
    
    def localSearch(self, n_iterations):
        """
        Performs localSearch to determine an optimised route to the Traveling Salesman Problem 
        Results are saved to a csv file for processing later

        Args:
            basePerm (list[int]): Initial tour
            n_iterations (striintng): The number of attempts the algorithm will have to produce an optimised value 

            """
        
        iteration_limit = 170_000
        
        for i in range(n_iterations):
            print(f"{i}:")
            
            # Generate random initial permutation
            base_perm = self.generate_random_path(self.dimension)
            
            # Calculate the overall cost
            base_cost = self.permutationCost(base_perm)
            
            checkpoint = time.perf_counter()
            
            # Calculate results for the jump
            print("Jump: ", end="")
            jump_cost = base_cost
            jump_perm = base_perm
            i = 0
            while True:
                if i >= iteration_limit:
                    break
                
                jump_perm, temp_cost = self.jump(jump_perm, jump_cost)
                if (temp_cost < jump_cost):
                    jump_cost = temp_cost
                else:
                    break
                    
                i += 1
                    
            
            # Output time taken
            print(f"{time.perf_counter() - checkpoint:.2f} seconds")
            checkpoint = time.perf_counter()


            # Calculate results for the exchange
            print("Exchange: ", end="")
            exch_cost = base_cost
            exch_perm = base_perm
            i = 0
            while True:
                if i >= iteration_limit:
                    break
                
                exch_perm, temp_cost = self.exchange(exch_perm, exch_cost)
                if (temp_cost < exch_cost):                    
                    exch_cost = temp_cost
                else:
                    break
                
                i += 1
                    
                        
            # Output time taken
            print(f"{time.perf_counter() - checkpoint:.2f} seconds")
            checkpoint = time.perf_counter()
            
            # Calculate results for the inversion
            print("Inversion: ", end="")
            invs_cost = base_cost
            invs_perm = base_perm
            i = 0
            while True:
                if i >= iteration_limit:
                    break
                
                invs_perm, temp_cost = self.inversion(invs_perm, invs_cost)
                if (temp_cost < invs_cost):                    
                    invs_cost = temp_cost
                else:
                    break
                
                i += 1
                

            # Output time taken
            print(f"{time.perf_counter() - checkpoint:.2f} seconds")
            print()


            self.saveData(jump_perm, jump_cost, exch_perm, exch_cost, invs_perm, invs_cost)


        
    def loadSaveFile(self, file_path):
        """
        Prepares the save file for the algorithm. It creates a new file if required, and loads it into memory. 
        If the file already exists, then a new file will be created with a slightly modified name (#)

        Args:
            filePath (string): file path to the csv file where the data will be saved. 

        Returns:
            """
        
        # if the file exists, give it a unique name so that it does not get overriden
        temp_file_name = file_path
        pos = len(file_path) - 4 # ignore the .csv at the end
        n = 1
        while os.path.exists(temp_file_name):
            temp_file_name = file_path[:pos] + '(' + n.__str__() + ')' + file_path[pos:]
            n += 1
        file_path = temp_file_name

        # Generate the file and the csv writer for use
        self.save_file = open(file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.save_file)
        self.csv_writer.writerow(['Jump - tour', 'Jump - cost', 'Exchange - tour', 'Exchange - cost', 'Inverse - tour', 'Inverse - cost'])
        self.save_file.flush()

    def saveData(self, jump_tour, jump_cost, exchange_tour, exchange_cost, inverse_tour, inverse_cost):
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
        self.csv_writer.writerow([jump_tour, jump_cost, exchange_tour, exchange_cost, inverse_tour, inverse_cost])
        self.save_file.flush()
        
    def getSavePath(self):
        return self.save_file.name
    