from tsp import TSP

# number of tests for each example
n = 30

EIL51 = TSP("final/code/test_cases/eli51", "final/code/temp_results/eil51.csv")
#EIL51.localSearch(n)
EIL51.saveData(1, 2, 3, 4, 5, 6)
EIL51.saveData(11, 12, 93, 54, 25, 66)
EIL51.saveData(12, 22, 33, 44, 65, 46)
EIL51.saveData(31, 2, 23, 46, 25, 67)

