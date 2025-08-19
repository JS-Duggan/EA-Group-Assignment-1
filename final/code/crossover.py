import random

"""
Crossover operators for permutation-encoded GAs (e.g., TSP routes).

Expected input for all methods:
- parent_1, parent_2: lists of unique, hashable items of equal length (a permutation).

Expected output for all methods:
- (child_1, child_2): two lists (same length as parents) representing valid offspring permutations.
"""


class Crossover:
    """Collection of crossover operators for permutations.

    Methods assume parents are permutations (no duplicates) and return two
    children of the same length. No validation is performed.
    """

    def __init__(self):
        return

    def order_crossover(self, parent_1, parent_2):
        """Order Crossover (OX).

        Inputs:
            parent_1, parent_2 (list): equal-length permutations.
        Returns:
            (child_1, child_2): offspring preserving relative order of
            non-segment elements from the opposite parent.
        """
        size = len(parent_1)

        # Select an arbitrary slice
        i, j = sorted(random.sample(range(size + 1), 2))

        # child 1
        child_1 = [0] * size                     # placeholder array
        child_1[i:j] = parent_1[i:j]             # copy fixed segment

        # Rotation of parent_2 starting at i to preserve order; remove copied items
        order = parent_2[-(i - 1):] + parent_2[:-(i - 1)]
        order = [item for item in order if item not in parent_1[i:j]]

        # Fill remaining positions
        index = j
        while True:
            if index >= size:
                index = 0
            if child_1[index] != 0:              # stop once we wrap back to the filled segment
                break
            child_1[index] = order.pop(0)
            index += 1

        # child 2
        child_2 = [0] * size
        child_2[i:j] = parent_2[i:j]

        order = parent_1[-(i - 1):] + parent_1[:-(i - 1)]
        order = [item for item in order if item not in parent_2[i:j]]

        index = j
        while True:
            if index >= size:
                index = 0
            if child_2[index] != 0:
                break
            child_2[index] = order.pop(0)
            index += 1

        return child_1, child_2

    def partially_mapped_crossover(self, parent_1, parent_2):
        """Partially Mapped Crossover (PMX).

        Inputs:
            parent_1, parent_2 (list): equal-length permutations.
        Returns:
            (child_1, child_2): offspring where a central segment is mapped
            between parents and the remainder keeps positional information.
        Notes:
            i, j are randomly chosen then overridden below (i=3, j=7), which
            appears intended for deterministic testing.
        """
        size = len(parent_1)

        # Select an arbitrary segment
        i, j = sorted(random.sample(range(size + 1), 2))

        # child 1
        child_1 = [0] * size
        child_1[i:j] = parent_1[i:j]            

        # Map non-common elements
        map = i
        for k in range(i, j):
            while parent_1[i:j].count(parent_2[k]) and k < j:
                k += 1
            if k >= j:
                break
            while parent_2[i:j].count(parent_1[map]) and map < j:
                map += 1
            if map >= j:
                break
            child_1[parent_2.index(parent_1[map])] = parent_2[k]
            map += 1

        # fill in place from p2
        index = j
        while child_1.count(0) > 0:
            if index >= size:
                index = 0
            if child_1[index] == 0:
                child_1[index] = parent_2[index]
            index += 1

        # Child 2
        child_2 = [0] * size
        child_2[i:j] = parent_2[i:j]

        # map non common elements
        map = i
        for k in range(i, j):
            while parent_2[i:j].count(parent_1[k]) and k < j:
                k += 1
            if k >= j:
                break
            while parent_1[i:j].count(parent_2[map]) and map < j:
                map += 1
            if map >= j:
                break
            child_2[parent_1.index(parent_2[map])] = parent_1[k]
            map += 1

        # Fill in place from p1
        index = j
        while child_2.count(0) > 0:
            if index >= size:
                index = 0
            if child_2[index] == 0:
                child_2[index] = parent_1[index]
            index += 1

        return child_1, child_2

    def cycle_crossover(self, parent_1, parent_2):
        """Cycle Crossover (CX).

        Inputs:
            parent_1, parent_2 (list): equal-length permutations.
        Returns:
            (child_1, child_2): offspring built by alternating cycles that
            map element positions between parents.
        """

        # Helper: return index of next unvisited position, or -1 if done
        def get_index(lst):
            for x in lst:
                if x != -1:
                    return x
            return -1

        size = len(parent_1)

        # Compute cycles for p1
        p1_cycles = []
        indices = list(range(0, size))           # tracks unvisited indices

        while True:
            cycle = [0] * size
            index = get_index(indices)
            if index == -1:
                break
            while True:
                if cycle[index] != 0:            # completed this cycle
                    break
                val = parent_1[index]
                cycle[index] = val
                indices[index] = -1
                index = parent_2.index(val)      # follow mapping into parent_2
            p1_cycles.append(cycle)

        # Compute cycles for p2
        p2_cycles = []
        indices = list(range(0, size))

        while True:
            cycle = [0] * size
            index = get_index(indices)
            if index == -1:
                break
            while True:
                if cycle[index] != 0:
                    break
                val = parent_2[index]
                cycle[index] = val
                indices[index] = -1
                index = parent_1.index(val)
            p2_cycles.append(cycle)


        # Alternate cycles to form children
        child_1 = [0] * size
        child_2 = [0] * size
        for i in range(len(p1_cycles)):
            for j in range(size):
                if i % 2 == 0:                    # even cycles -> p1 to child_1, p2 to child_2
                    for k in range(size):
                        if p1_cycles[i][k] != 0:
                            child_1[k] = p1_cycles[i][k]
                        if p2_cycles[i][k] != 0:
                            child_2[k] = p2_cycles[i][k]
                else:                             # odd cycles -> swap
                    for k in range(size):
                        if p1_cycles[i][k] != 0:
                            child_2[k] = p1_cycles[i][k]
                        if p2_cycles[i][k] != 0:
                            child_1[k] = p2_cycles[i][k]

        return child_1, child_2
