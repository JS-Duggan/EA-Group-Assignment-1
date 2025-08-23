
from __future__ import annotations
import random
from typing import List, Tuple, Optional
import numpy as np

from load_tsp import loadTSP


def tour_cost(graph: np.ndarray, tour: List[int]) -> float:
    """Compute cyclic tour cost using the given distance matrix."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += graph[tour[i], tour[(i + 1) % n]]
    return float(total)


def neighbors_in_tour(tour: List[int], city: int) -> Tuple[int, int]:
    """Return (prev, next) city of `city` in the current tour."""
    n = len(tour)
    idx = tour.index(city)
    return tour[(idx - 1) % n], tour[(idx + 1) % n]


def invert_segment_circular(tour: List[int], start_after_idx: int, j_idx: int) -> None:
    """
    Invert the segment from index (start_after_idx) to index j_idx (inclusive) in circular sense.
    If start_after_idx <= j_idx -> normal slice reverse.
    Else handle wrap-around by reversing the concatenated segment and writing back.
    """
    n = len(tour)
    i = start_after_idx
    j = j_idx
    if i == j:
        return
    if i < j:
        tour[i:j + 1] = reversed(tour[i:j + 1])
    else:
        # Wrap-around: segment = tour[i:] + tour[:j+1]
        seg = tour[i:] + tour[:j + 1]
        seg.reverse()
        k = 0
        m1 = n - i
        tour[i:] = seg[:m1]
        k += m1
        tour[:j + 1] = seg[k:]


class InverOverEA:
    """
    Inver-over EA (Tao & Michalewicz):
      - Representation: permutation of cities [0..n-1]
      - Parent-offspring competition
      - Operator: guided inversion with prob 1-p; random with prob p (p ~ 0.02)
      - Stop when the proposed next city is already adjacent in the child
    """
    def __init__(self, graph: np.ndarray, p_random: float = 0.02,
                 population_size: int = 50, generations: int = 20000,
                 seed: Optional[int] = None):
        self.graph = graph
        self.n = graph.shape[0]
        self.p = p_random
        self.pop_size = population_size
        self.generations = generations
        if seed is not None:
            random.seed(seed)

    def _random_perm(self) -> List[int]:
        tour = list(range(self.n))
        random.shuffle(tour)
        return tour

    def _offspring_inver_over(self, parent: List[int], population: List[List[int]]) -> List[int]:
        """Single offspring via Inver-over as per the paper."""
        n = self.n
        child = parent[:]  # work on copy
        c = random.choice(child)
        while True:
            # choose c0
            if random.random() <= self.p:
                # random c0 != c
                candidates = [x for x in child if x != c]
                c0 = random.choice(candidates)
            else:
                # guided by a random mate: c0 = successor of c in mate
                mate = random.choice(population)
                idx = mate.index(c)
                c0 = mate[(idx + 1) % n]

            prev_c, next_c = neighbors_in_tour(child, c)
            if c0 == prev_c or c0 == next_c:
                break  # stop condition

            i = child.index(c)
            i_next = (i + 1) % n
            j = child.index(c0)
            invert_segment_circular(child, i_next, j)
            c = c0
        return child

    def run(self) -> Tuple[List[int], float]:
        """Run EA with parent-offspring competition per individual for `generations`."""
        # init population
        pop = [self._random_perm() for _ in range(self.pop_size)]
        costs = [tour_cost(self.graph, t) for t in pop]
        best_idx = int(min(range(self.pop_size), key=lambda i: costs[i]))
        best = pop[best_idx][:]
        best_cost = costs[best_idx]

        for _ in range(self.generations):
            for i in range(self.pop_size):
                parent = pop[i]
                child = self._offspring_inver_over(parent, pop)
                child_cost = tour_cost(self.graph, child)
                if child_cost <= costs[i]:
                    pop[i] = child
                    costs[i] = child_cost
                    if child_cost < best_cost:
                        best_cost = child_cost
                        best = child[:]
        return best, float(best_cost)


def run_on_instance(path: str, runs: int = 30, population_size: int = 50, generations: int = 20000,
                    p_random: float = 0.02, seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Load TSPLIB file via your loadTSP class, run Inver-over EA `runs` times,
    return (mean, stddev) of tour costs.
    """
    data = loadTSP(path)
    graph = data.get_distance_matrix()
    # n = data.get_dimension()  # not used directly here

    costs: List[float] = []
    base_seed = seed if seed is not None else int(random.random() * 1e9)
    for r in range(runs):
        ea = InverOverEA(graph, p_random=p_random, population_size=population_size,
                         generations=generations, seed=base_seed + r * 991)
        _, cost = ea.run()
        costs.append(cost)

    mean = float(sum(costs) / len(costs))
    # sample stddev over the 30 runs
    if len(costs) > 1:
        m = mean
        var = sum((c - m) ** 2 for c in costs) / (len(costs) - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std
