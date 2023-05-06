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

def create_population(pop_len, func):
    population = []
    for _ in range(pop_len):
        population.append(create_ind())
    first_pop_values = fitness_func(population, func)
    return population, first_pop_values

def fitness_func(population, func):
    fitness = []
    for ind in population:
        fitness.append(func(ind))
    return fitness

def parents(parents, fitness):
    tournament = []
    while len(tournament) != 3:
        parent = random.choice(parents)
        tournament.append(fitness[parents.index(parent)])
    winner = parents[fitness.index(min(fitness))]
    return winner

def cross(parent1, parent2, cross_prob):
    if random.uniform(0, 1) < cross_prob:
        child1 = [parent1[0],parent2[1]]
        child2 = [parent2[0],parent1[1]]
        return child1, child2
    else:
        return parent1, parent2

def mutation(new_gen, mut_prob):
    for i in range(2):
        for j in range(2):
            if random.random() < mut_prob:
                new_gen[i][j] += np.random.random()

def genetic_alg(func, num_gen=50, pop_len=100, cross_prob=0.8, mut_prob=0.1):
    population, fitness_values = create_population(pop_len, func)
    for i in range(num_gen):
        offspting = []
        for j in range(int(len(population)/2)):
            parent1 = parents(population, fitness_values)
            parent2 = parents(population, fitness_values)
            child1, child2 = cross(parent1, parent2, cross_prob)
            mutation([child1, child2], mut_prob)
            offspting.append(child1)
            offspting.append(child2)
        fitness_offspring = fitness_func(offspting, func)
        avg_fitness = sum(fitness_offspring) / len(fitness_offspring)
        print(i, "  ", fitness_offspring[0], " ", avg_fitness)

genetic_alg(matyas)