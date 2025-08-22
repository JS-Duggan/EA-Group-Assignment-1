from Individual import Individual
from tsp import TSP

class Population:
    def __init__(self, pop_size, tsp: TSP):
        self.individuals = []
        for _ in range(pop_size):
            gen_ind = Individual(tsp)
            self.individuals.append(gen_ind)
    
    def get_population(self):
        all_paths = []
        for gen_ind in self.individuals:
            all_paths.append(gen_ind.get_path())
        return all_paths