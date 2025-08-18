import random

class Crossover:
    def __init__(self):
        return

    def order_crossover(self, parent_1, parent_2):

        size = len(parent_1)
        # select arbitrary segment
        i, j = sorted(random.sample(range(size + 1), 2))
        # Perform for parent 1
        #   copy segment to child
        child_1 = [0] * size
        child_1[i:j] = parent_1[i:j]

        #   create ordered list of elements to insert
        order = parent_2[-(i-1):] + parent_2[:-(i-1)]
        order = [item for item in order if item not in parent_1[i:j]]

        index = j
        #   copy remaining alleles, using order of parent 2
        while True:
            if index >= size:
                index = 0
            if child_1[index] != 0:
                break
            child_1[index] = order.pop(0)
            index += 1

        # Perform for parent 2
        child_2 = [0] * size
        #   copy segment to child
        child_2[i:j] = parent_2[i:j]

        #   create ordered list of elements to insert
        order = parent_1[-(i-1):] + parent_1[:-(i-1)]
        order = [item for item in order if item not in parent_2[i:j]]

        index = j
        #   copy remaining alleles, using order of parent 2
        while True:
            if index >= size:
                index = 0
            if child_2[index] != 0:
                break
            child_2[index] = order.pop(0)
            index += 1
        #   copy remaining alleles, using order of parent 2

        return child_1, child_2