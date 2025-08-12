from tsp import TSP
from random_path import randomPath

testPath = 'final/code/test_cases/eil51.tsp'
savePath = 'final/code/saves/save.txt'

tsp = TSP(testPath, savePath)

random_path = randomPath()
dimension = tsp.getDimension()
permutation = randomPath.generate_random_path(dimension)

print(permutation)

nIterations = 30
tsp.localSearch(permutation, 30)

