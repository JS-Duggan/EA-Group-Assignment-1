
# Exercise 1-2

A greedy approach to the TSP problem by using one of 3 possible mutation operators. Each itteration, it randomly selects one possible mutation in the neighbourhood and checks to see if it improves the cost. If so, the improvement is made and the itteration is finished, otherwise it keeps checking through every possible itteration at random. The algorithm finishes when there are no neighbours that improve the permutation or when 170,000 itterations have been processed. 

To run the code:

`python final/code/main.py tsp_instance iterations`
- `tsp_instance` is one of the data sets. E.g. `eil51`
- `iterations` is the number of itteration attempts to run the entire algorithm. E.g. `30`

The data is collated into a csv file under `final/code/saves` as a raw set of data. A statistical summary is also displayed in the terminal.

# Exercise 3-6


# Exercise 7

This aglorithm looks to impliment the Inver-Over Algorithm. 

To run the code:

`final/code/run_inverover.py`

Arguments
- `--runs` 
  - Number of times the algorithm will run.
  - type: `int`
  - default: `30`
- `--pop` 
  - Population size of the evolutionary algorithm.
  - type: `int`
  - default: `50`
- `--gens` 
  - The number of generations that will be evalutated.
  - type: `int`
  - default: `20000`
- `--p_random` 
  - The chance of a random mutation occuring.
  - type: `float` 
  - default: `0.02`
- `--out`
  - The output file of the data.
  - type: `str`
  - default: `results/inverover.txt`
- `--seed`
  - A value to seed the initial generation of the algorithm.
  - type: `int`
  - default: `None`
