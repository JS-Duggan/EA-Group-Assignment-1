import argparse

from tsp import TSP
from random_path import randomPath

# Parse input arguments:
"""
Run in terminal: python main.py tsp_instance iterations

tsp_instance: e.g., 'eil51'
iterations: e.g., 30
"""
parser = argparse.ArgumentParser()
parser.add_argument('tsp_instance')
parser.add_argument('iterations')
args = parser.parse_args()
tsp_instance = args.tsp_instance
iterations = int(args.iterations)

testPath = f'final/code/test_cases/{tsp_instance}.tsp'
savePath = f'final/code/saves/{tsp_instance}_out.csv'

tsp = TSP(testPath, savePath)

random_path = randomPath()
dimension = tsp.getDimension()
permutation = randomPath.generate_random_path(dimension)

# Run local search
tsp.localSearch(permutation, iterations)

