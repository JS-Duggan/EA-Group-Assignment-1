import random

class Selection:
    def __init__(self):
        return


    # Fitness Selection Roulette Wheel
    def fitnessProportional(self, population, fitnesses):
        """
        Expected copies of i: μ * f(i) / Σf
        Spin a 1-armed wheel n times

        Inputs:
            population (list[int]): Size of the generated popultation
            fitness (list[int]): list of fitnesses in the popultaion
        """
        size = len(population)
        total_fitness = sum(fitnesses)
        probs = [f / total_fitness for f in fitnesses]

        selected = []
        for _ in range(size):
            r = random.random()
            cumulative = 0
            for i in range(size):
                cumulative += probs[i]
                if r <= cumulative:
                    selected.append(population[i])
                    break
        return selected


    def tournament(self, population, fitnesses, k=3, p=1.0):
        """
        Pick k members at random
        Select best with probability p, else random contestant
        """
        size = len(population)
        selected = []

        for _ in range(size):
            # pick k random contestants
            contestants_idx = random.sample(range(size), k)
            contestants = [(population[i], fitnesses[i]) for i in contestants_idx]

            # sort contestants by fitness
            contestants.sort(key=lambda x: x[1], reverse=True)

            # best wins with prob p
            if random.random() < p:
                winner = contestants[0][0]
            else:
                winner = random.choice(contestants)[0]

            selected.append(winner)
        return selected

    
    def elitism(self, population, fitnesses, num_elites=1):
        """
        Copy the top individuals directly into the next generation
        """
        size = len(population)
        # pair (fitness, individual) and sort descending
        paired = list(zip(fitnesses, population))
        paired.sort(key=lambda x: x[0], reverse=True)

        elites = []
        for i in range(num_elites):
            elites.append(paired[i][1])  # take best individuals

        return elites
