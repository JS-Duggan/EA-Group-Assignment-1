import random

def generate_random_path(num_nodes):
    path = list(range(1, num_nodes + 1))
    random.shuffle(path)
    return path
