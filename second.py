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
    return [random.uniform(-5,5), random.uniform(-5,5)]

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
        tournament.append(fitness[parents.index(parent)])
    winner = parents[fitness.index(min(fitness))]
    return winner

def cross(parent1, parent2, cross_prob):
    if random.random() < cross_prob:
        child1 = [parent1[0], parent2[1]]
        child2 = [parent2[0], parent1[1]]
        return child1, child2
    else:
        return parent1, parent2

def mutation(new_gen, mut_prob):
    for i in range(2):
        for j in range(2):
            if random.random() < mut_prob:
                new_gen[i][j] += random.uniform(-1, 1)

def genetic_alg(func, num_gen=50, pop_len=100, cross_prob=0.9, mut_prob=0.1):
    population = create_population(pop_len)
    fitness_values = fitness_func(population, func)
    min_val = []
    avg_val = []
    for i in range(num_gen):
        offspring = []
        for _ in range(int(len(population)/2)):
            parent1 = parents(population, fitness_values)
            parent2 = parents(population, fitness_values)
            child1, child2 = cross(parent1, parent2, cross_prob)
            mutation([child1, child2], mut_prob)
            offspring.append(child1)
            offspring.append(child2)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        min_val.append(min(fitness_values))
        avg_val.append(avg_fitness)
        print("Gen:", i+1, " Min:",  min(fitness_values), " Avg:", avg_fitness)
        population = offspring
        fitness_values = fitness_func(population, func)
    avg_val = sorted(avg_val, reverse=True)
    min_coord = population[fitness_values.index(min(fitness_values))]
    return min_val, avg_val, min_coord

def draw(func):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            Z[i, j] = func([X[i, j], Y[i, j]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if func == bukin6:
        ax.set_title('Bukin Function 6')
    elif func == matyas:
        ax.set_title('Matyas Function')
    elif func == schaffer2:
        ax.set_title('Schaffer Function 2')
    plt.show()

def main_func(func):
    minv, avgv, min_coord = genetic_alg(func)
    plt.plot(minv, label="min")
    plt.plot(avgv, label="avg")
    plt.legend()
    plt.show()
    print("[x;y] = ",min_coord)
    print("f(x,y) = ", func(min_coord))

print(" Matyas")
main_func(matyas)
draw(matyas)
print("\n Bukin 6")
main_func(bukin6)
draw(bukin6)
print("\n Schaffer 2")
main_func(schaffer2)
draw(schaffer2)

