import multiprocessing
from multiprocessing import Pool
from multiprocessing import shared_memory
import numpy as np
import argparse

from tsp import TSP
from load_tsp import loadTSP
from evolution import EvolutionaryAlgorithm

# Each parallel process runs
def wrapper(args):
    """
    runs the evolition algorithm on a set of arguments
    Args:
        args (): argments for the function
    Returns
        mean (float): The mean of the results
        std (float): The standard deviation of the results
    """
    shm_name, shape, dtype, runs, population, generations, dimension, path = args
    
    tsp = TSP(path, "dummy_save.csv", dimension, False)
    tsp.load_shared_memory(shm_name, shape, dtype)
    
    ea = EvolutionaryAlgorithm(tsp, population)
    
    ea.initialize_population()
    population = ea.population
    
    results = []
    for i in range(runs):
        results.append(ea.exploitation(population, generations, tsp)) # EA function
        
    return results

def main():
    """
    The main function which runs the EA algorithm for a set TSP instance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('tsp_instance')
    parser.add_argument('population')
    parser.add_argument('generations')
    parser.add_argument('iterations')
    args = parser.parse_args()
    tsp_instance = args.tsp_instance
    population = int(args.population)
    generations = int(args.generations)
    iterations = int(args.iterations)
    
    path = f'final/code/test_cases/{tsp_instance}.tsp'
    
    # Number of parallel processes
    num_workers = multiprocessing.cpu_count()
    runs_per_worker = iterations // num_workers
    extra_runs = iterations % num_workers
    
    data = loadTSP(path)
    graph = data.get_distance_matrix()
    
    dimension = data.get_dimension()
    
    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=graph.nbytes)
    shared_graph = np.ndarray(graph.shape, dtype=graph.dtype, buffer=shm.buf)
    shared_graph[:] = graph[:]  # copy data
    del graph
    del data
    
    # Arguments passed to each process
    all_tasks = []
    for i in range(num_workers):
        runs = runs_per_worker + (1 if i < extra_runs else 0)
        if runs > 0:
            all_tasks.append((shm.name, shared_graph.shape, shared_graph.dtype,
                              runs,
                              population,
                              generations,
                              dimension,
                              path
            ))
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(wrapper, all_tasks)
    
    res = []
    for i in range(len(results)):
        for j in range(len(results[i])):
            res.append(results[i][j][1])
        
    
    
    mean = np.mean(res)
    std = np.std(res)
    
    print(f"Runs: {len(res)}")
    print(f"- mean: {mean}")
    print(f"- std: {std}")
    
    
    # Cleanup
    shm.close()
    shm.unlink()

if __name__ == "__main__":
    main()