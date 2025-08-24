
from __future__ import annotations
import argparse, glob, time
from pathlib import Path
from inverover import run_on_instance
import multiprocessing
from multiprocessing import Pool
from multiprocessing import shared_memory
from collections import defaultdict
import numpy as np

from load_tsp import LoadTSP

def wrapper(args):
    """
    runs the inverover algorithm on a set of arguments
    Args:
        args (): argments for the function
    Returns
        mean (float): The mean of the results
        std (float): The standard deviation of the results
    """
    return run_on_instance(*args)

def main():
    """
    Processes the input and runs the inverover algorithm on the specified TSP instance
    """
    ap = argparse.ArgumentParser(
        description="Run Inver-over EA (Tao & Michalewicz) on TSPLIB instances and write results/inverover.txt"
    )
    ap.add_argument("--instances_glob", type=str, required=True,
                    help='Glob for .tsp files, e.g. "tsplib/*.tsp"')
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--pop", type=int, default=50)
    ap.add_argument("--gens", type=int, default=20000)
    ap.add_argument("--p_random", type=float, default=0.02)
    ap.add_argument("--out", type=str, default="results/inverover.txt")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.instances_glob))
    if not files:
        raise SystemExit(f"No instances matched: {args.instances_glob}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    num_workers = multiprocessing.cpu_count()
    # num_workers = 2
    runs_per_worker = args.runs // num_workers
    extra_runs = args.runs % num_workers

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Inver-over EA results (mean, stddev over runs)\n")
        f.write(f"# runs={args.runs}, pop={args.pop}, gens={args.gens}, p_random={args.p_random}\n")
        f.write("instance,mean_cost,stddev\n")
        
        all_tasks = []
        for p in files:
            # Load graph once
            data = LoadTSP(p)
            graph = data.get_distance_matrix()
            
            # Create shared memory
            shm = shared_memory.SharedMemory(create=True, size=graph.nbytes)
            shared_graph = np.ndarray(graph.shape, dtype=graph.dtype, buffer=shm.buf)
            shared_graph[:] = graph[:]  # copy data

            del graph
            del data
            
            for i in range(num_workers):
                runs = runs_per_worker + (1 if i < extra_runs else 0)
                if runs > 0:
                    all_tasks.append((shm.name, shared_graph.shape, shared_graph.dtype, runs, args.pop, args.gens, args.p_random, args.seed, 50))
        
        with Pool(processes=num_workers) as pool:
            partial_results = pool.map(wrapper, all_tasks)
            
        results_per_instance = defaultdict(list)
        # Collect results by instance
        for (mean, std), task in zip(partial_results, all_tasks):
            p = task[0]
            results_per_instance[p].append(mean)

            
        for p in files:
            means = results_per_instance[p]
            mean_final = np.mean(means)
            std_final = np.std(means)
            
            name = Path(p).name
            f.write(f"{name},{mean_final:.6f},{std_final:.6f}\n")
            f.flush()
            print(f"[DONE] {name}: mean={mean_final:.2f} std={std_final:.2f}", flush=True)


    print(f"Wrote {out_path}")
    
    # Cleanup
    shm.close()
    shm.unlink()

if __name__ == "__main__":
    main()
