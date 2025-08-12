import random

def generate_random_path(num_nodes):
    path = list(range(0, num_nodes))
    random.shuffle(path)
    return path

