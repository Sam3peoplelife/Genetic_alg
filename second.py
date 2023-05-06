import random
import math
import matplotlib.pyplot as plt
import numpy as np

def bukin6(x):
    return 100 * math.sqrt(abs(x[1] - 0.01 * x[0]**2)) + 0.01 * abs(x[0] + 10)

def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def schaffer2(x):
    return 0.5 + ((math.sin(x[0]**2 - x[1]**2)**2 - 0.5) / (1 + 0.001 * (x[0]**2 + x[1]**2))**2)

def create_ind():
    return [random.uniform(-10,10), random.uniform(-10,10)]

def create_population(pop_len):
    population = []
    for _ in range(pop_len):
        population.append(create_ind())
    return population

def fitness_func(population, func):
    fitness = []
    for ind in population:
        fitness.append(func(ind))
    return fitness

def parents(parents, fitness):
    tournament = []
    while len(tournament) != 3:
        parent = random.choice(parents)
        tournament.append(fitness[parent.index()])
    winner = parents[fitness.index(max(fitness))]
    return winner

def cross(parent1, parent2, cross_prob):
    if random.uniform(0, 1) < cross_prob:
        child1 = [parent1[0],parent2[1]]
        child2 = [parent2[0],parent1[1]]
        return child1, child2
    else:
        return parent1, parent2

def mutation(new_gen, mut_prob):
    for i in new_gen:
        for coord in i:
            if random.uniform(0, 1) < mut_prob:
                new_gen[coord] += np.random.random()

