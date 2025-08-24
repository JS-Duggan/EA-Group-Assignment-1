import argparse

import data_summary
from tsp import TSP

"""
Runs the TSP algorithm for Exercise 2 on a particular TSP instance

Run in terminal: python main.py tsp_instance iterations

tsp_instance: e.g., 'eil51'
iterations: e.g., 30
"""

# Parse input arguments:
parser = argparse.ArgumentParser()
parser.add_argument('tsp_instance')
parser.add_argument('iterations')
args = parser.parse_args()
tsp_instance = args.tsp_instance
iterations = int(args.iterations)

test_path = f'final/code/test_cases/{tsp_instance}.tsp'
save_path = f'final/code/saves/{tsp_instance}.csv'

tsp = TSP(test_path, save_path)

# Run local search
tsp.localSearch(iterations)

# Output data summary
data_summary.processData(tsp.getSavePath())
