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
        #   copy remaining alleles, using order of parent 1
        while True:
            if index >= size:
                index = 0
            if child_2[index] != 0:
                break
            child_2[index] = order.pop(0)
            index += 1

        return child_1, child_2
    
    def partially_mapped_crossover(self, parent_1, parent_2):
        size = len(parent_1)
        # select arbitrary segment
        i, j = sorted(random.sample(range(size + 1), 2))
        i, j = 3, 7
        # Perform for parent 1
        #   copy segment to child
        child_1 = [0] * size
        child_1[i:j] = parent_1[i:j]
        map = i
        #   map non-common elements
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

        #   copy remaining elements in place
        index = j
        while child_1.count(0) > 0:
            if index >= size:
                index = 0
            if child_1[index] == 0:
                child_1[index] = parent_2[index]
            index += 1


        # Perform for parent 2
        #   copy segment to child
        child_2 = [0] * size
        child_2[i:j] = parent_2[i:j]
        map = i
        for k in range(i, j):
            while parent_2[i:j].count(parent_1[k]) and k < j:
                k += 1
            if k >= j:
                break
            print(parent_1[k])
            while parent_1[i:j].count(parent_2[map]) and map < j:
                map += 1
            if map >= j:
                break
            child_2[parent_1.index(parent_2[map])] = parent_1[k]
            map += 1

        index = j
        while child_2.count(0) > 0:
            if index >= size:
                index = 0
            if child_2[index] == 0:
                child_2[index] = parent_1[index]
            index += 1

        return child_1, child_2