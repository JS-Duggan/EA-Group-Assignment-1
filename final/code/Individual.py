from tsp import TSP

class Individual:
    def generate_single_permutation(self, num_nodes):
        individual_perm = TSP()
        individual_perm.generate_random_path(self, num_nodes)
        return individual_perm
    

