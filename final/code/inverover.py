from __future__ import annotations
import random
from typing import List, Tuple, Optional
import numpy as np

from load_tsp import loadTSP

def tour_cost(graph: np.ndarray, tour: List[int]) -> float:
    """Compute cyclic tour cost."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += graph[tour[i], tour[(i + 1) % n]]
    return float(total)

def make_pos(tour: List[int]) -> List[int]:
    """pos[city] = index of city in tour."""
    pos = [0] * len(tour)
    for i, c in enumerate(tour):
        pos[c] = i
    return pos

def invert_segment_circular_with_pos(tour: List[int], pos: List[int], start_idx: int, end_idx: int) -> None:
    """
    Reverse the segment from start_idx to end_idx (inclusive) on the circular tour.
    Updates 'pos' accordingly.
    """
    n = len(tour)
    if start_idx == end_idx:
        return
    if start_idx < end_idx:
        # straight slice
        seg = tour[start_idx:end_idx + 1]
        seg.reverse()
        tour[start_idx:end_idx + 1] = seg
        for k in range(start_idx, end_idx + 1):
            pos[tour[k]] = k
    else:
        # wrap-around: [start_idx..n-1] + [0..end_idx]
        seg = tour[start_idx:] + tour[:end_idx + 1]
        seg.reverse()
        m1 = n - start_idx
        tour[start_idx:] = seg[:m1]
        tour[:end_idx + 1] = seg[m1:]
        for k in range(start_idx, n):
            pos[tour[k]] = k
        for k in range(0, end_idx + 1):
            pos[tour[k]] = k

class InverOverEA:
    """
    Faster Inver-over EA (Tao & Michalewicz) with:
      - O(1) adjacency using position arrays
      - O(1) delta cost per inversion (symmetric TSP)
      - optional progress printing
    """
    def __init__(self,
                 graph: np.ndarray,
                 p_random: float = 0.02,
                 population_size: int = 50,
                 generations: int = 20000,
                 seed: Optional[int] = None,
                 progress_every: int = 0):
        self.graph = graph
        self.n = graph.shape[0]
        self.p = p_random
        self.pop_size = population_size
        self.generations = generations
        self.progress_every = progress_every
        if seed is not None:
            random.seed(seed)

    def _random_perm(self) -> List[int]:
        t = list(range(self.n))
        random.shuffle(t)
        return t

    def _offspring_inver_over(self,
                              parent: List[int], pos_parent: List[int],
                              population: List[List[int]], pop_pos: List[List[int]],
                              parent_cost: float) -> Tuple[List[int], List[int], float]:
        """Produce one child by Inver-over with delta-cost and position maintenance."""
        n = self.n
        g = self.graph

        # Work on a copy
        child = parent[:]
        pos = pos_parent[:]  # shallow copy is fine (list of ints)
        cost = parent_cost

        # pick current city c
        c = random.choice(child)
        i = pos[c]

        while True:
            # choose c0
            if random.random() <= self.p:
                # random c0 != c
                # uniformly pick an index != i
                j_idx = i
                while j_idx == i:
                    j_idx = random.randrange(n)
                c0 = child[j_idx]
            else:
                # guided by random mate: successor of c in that mate
                k = random.randrange(self.pop_size)
                mate = population[k]
                mate_pos = pop_pos[k]
                j_in_mate = mate_pos[c]               # O(1)
                c0 = mate[(j_in_mate + 1) % n]
                j_idx = pos[c0]                        # index of c0 in child (O(1))

            # check adjacency of c and c0 in child
            prev_c = child[(i - 1) % n]
            next_c = child[(i + 1) % n]
            if c0 == prev_c or c0 == next_c:
                break  # stop condition

            # boundaries before inversion
            next_idx = (i + 1) % n
            after_c0_idx = (j_idx + 1) % n
            next_city = child[next_idx]
            after_c0 = child[after_c0_idx]

            # delta cost for reversal [next_idx .. j_idx] (circular):
            # cut (c,next) and (c0,after_c0); add (c,c0) and (next,after_c0)
            delta = - g[c, next_city] - g[c0, after_c0] + g[c, c0] + g[next_city, after_c0]
            cost += float(delta)

            # perform inversion with pos updates
            invert_segment_circular_with_pos(child, pos, next_idx, j_idx)

            # advance: new current city becomes c0
            c = c0
            i = pos[c]

        return child, pos, cost

    def run(self) -> Tuple[List[int], float]:
        # init population, positions, costs
        pop = [self._random_perm() for _ in range(self.pop_size)]
        pop_pos = [make_pos(t) for t in pop]
        costs = [tour_cost(self.graph, t) for t in pop]

        best_idx = min(range(self.pop_size), key=lambda i: costs[i])
        best = pop[best_idx][:]
        best_cost = costs[best_idx]

        for gen in range(1, self.generations + 1):
            for i in range(self.pop_size):
                parent = pop[i]
                parent_pos = pop_pos[i]
                parent_cost = costs[i]

                child, child_pos, child_cost = self._offspring_inver_over(
                    parent, parent_pos, pop, pop_pos, parent_cost
                )

                if child_cost <= parent_cost:
                    pop[i] = child
                    pop_pos[i] = child_pos
                    costs[i] = child_cost
                    if child_cost < best_cost:
                        best_cost = child_cost
                        best = child[:]

            if self.progress_every and (gen % self.progress_every == 0):
                print(f"[inver-over] gen {gen}/{self.generations} best={best_cost:.2f}", flush=True)

        return best, float(best_cost)

def run_on_instance(path: str, runs: int = 30, population_size: int = 50, generations: int = 20000,
                    p_random: float = 0.02, seed: Optional[int] = None, progress_every: int = 0) -> Tuple[float, float]:
    """Load TSPLIB via your loader, run Inver-over `runs` times, return (mean, stddev)."""
    data = loadTSP(path)
    graph = data.get_distance_matrix()

    costs: List[float] = []
    base_seed = seed if seed is not None else int(random.random() * 1e9)
    for r in range(runs):
        ea = InverOverEA(graph,
                         p_random=p_random,
                         population_size=population_size,
                         generations=generations,
                         seed=base_seed + r * 9973,
                         progress_every=generations // 10 if generations >= 10 else 0)
        _, cost = ea.run()
        costs.append(cost)

    mean = float(sum(costs) / len(costs))
    if len(costs) > 1:
        m = mean
        var = sum((c - m) ** 2 for c in costs) / (len(costs) - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std
