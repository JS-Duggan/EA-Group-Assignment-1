import argparse
import multiprocessing
from multiprocessing import Pool
from multiprocessing import shared_memory
from collections import defaultdict
import numpy as np

import runDataSummary
from tsp import TSP
from load_tsp import loadTSP

"""
Run in terminal: python main.py tsp_instance iterations

tsp_instance: e.g., 'eil51'
iterations: e.g., 30
"""

_tsp = None

def init_worker(shm_name, shape, dtype, save_path):
    global _tsp
    _tsp = TSP(shm_name, shape, dtype, save_path)  # attach to shared memory

def run_local_search(args):
    global _tsp
    return _tsp.localSearch(*args)


# def run_local_search(args):
#     # shm_name, shape, dtype, runs = args
#     tsp = TSP(shm.name, shared_graph.shape, shared_graph.dtype, savePath)
    
#     return tsp.localSearch(*args)


def main():
    # Parse input arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('tsp_instance')
    parser.add_argument('iterations')
    args = parser.parse_args()
    tsp_instance = args.tsp_instance
    iterations = int(args.iterations)

    testPath = f'final/code/test_cases/{tsp_instance}.tsp'
    savePath = f'final/code/saves/{tsp_instance}.csv'

    # Run local search in parallel
    num_workers = multiprocessing.cpu_count()
    runs_per_worker = iterations // num_workers
    extra_runs = iterations % num_workers

    # Load graph once
    data = loadTSP(testPath)
    graph = data.get_distance_matrix()

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=graph.nbytes)
    shared_graph = np.ndarray(graph.shape, dtype=graph.dtype, buffer=shm.buf)
    shared_graph[:] = graph[:]  # copy data
    dimension = data.get_dimension()
    del graph
    del data


    args_list = []
    for i in range(num_workers):
        runs = runs_per_worker + (1 if i < extra_runs else 0)
        if runs > 0:
            # args_list.append((shm.name, shared_graph.shape, shared_graph.dtype, runs))
            args_list.append((runs, dimension))
            
    with Pool(processes=num_workers,
              initializer=init_worker,
              initargs=(shm.name, shared_graph.shape, shared_graph.dtype, savePath)) as pool:
        results = pool.map(run_local_search, args_list)


    # Output data summary
    runDataSummary.processData(savePath)
    
    # Cleanup
    shm.close()
    shm.unlink()


if __name__ == "__main__":
    main()