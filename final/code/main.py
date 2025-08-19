import argparse

import runDataSummary
from tsp import TSP

"""
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

testPath = f'final/code/test_cases/{tsp_instance}.tsp'
savePath = f'final/code/saves/{tsp_instance}.csv'

tsp = TSP(testPath, savePath)

# Run local search
tsp.localSearch(iterations)

# Output data summary
runDataSummary.processData(tsp.getSavePath())
