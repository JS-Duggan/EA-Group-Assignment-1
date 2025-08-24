
# Exercise 1-2

A greedy approach to the TSP problem by using one of 3 possible mutation operators. Each itteration, it randomly selects one possible mutation in the neighbourhood and checks to see if it improves the cost. If so, the improvement is made and the itteration is finished, otherwise it keeps checking through every possible itteration at random. The algorithm finishes when there are no neighbours that improve the permutation or when 170,000 itterations have been processed. 

To run the code:

`python final/code/main.py tsp_instance iterations`
- `tsp_instance` is one of the data sets. E.g. `eil51`
- `iterations` is the number of itteration attempts to run the entire algorithm. E.g. `30`

The data is collated into a csv file under `final/code/saves` as a raw set of data. A statistical summary is also displayed in the terminal.

# Exercise 3-6

To run the code:

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Run the comprehensive evolution comparison:
   ```bash
   python final/code/run_evolutions.py {test_case}
   ```

  This will run all three algorithms with a population size of 20, and will save metrics for generations 2000, 5000, 10,000, and 20,000
  Results are saved to CSV files in `final/code/saves/` with detailed performance metrics including cost, runtime, improvement, and population diversity.

Running the best performing EA (exploitation) 30 times on each TSP instance:
  To run the exploitation algorithm 30 times, run `run_ea.py`.
  To run it in terminal (make sure you are in project root directory):
      `python final/code/run_ea.py tsp_instance population_size generations iterations`
          e.g.,: `python final/code/run_ea.py eil51 50 20000 30`

# Exercise 7

This aglorithm looks to impliment the Inver-Over Algorithm. 

To run the code with default arguments (make sure you are in final/code using `cd final/code`):
  `python run_inverover.py --instances_glob tsp_path`
      e.g.,: `python run_inverover.py --instances_glob test_cases/eil51.tsp`

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
