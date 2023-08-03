#!/bin/python

import random
import pandas as pd
import math
from scipy.stats import norm
import numpy as np
from scipy.integrate import quad
from deap import base, creator, tools, algorithms

file_data = 'Data_X.xlsx'
df = pd.read_excel (file_data)
X_prev = df['X']  # Specify X_(i-1) value

# trace 1
# print(X_prev)

Lu = 10  # Specify Lu value
rho = 0.3  # Specify rho value
mu = 0  # Specify mu value
sigma = 1  # Specify sigma value
n = 10  # Specify the number of cycles

def c(Lu, mu, sigma):
    # Compute probability using CDF of standard normal distribution
    return 1 - norm.cdf((Lu - mu) / sigma)

def f(x):
    # Compute probability using PDF of standard normal distribution
    return norm.pdf(x)

def L(Lu, rho, mu, sigma):
    # Compute lambda using the probability values
    return c(Lu, rho * (X_prev - mu) - mu, math.sqrt((1 - rho**2) * sigma**2))

def compute_p_i(Lu, rho, mu, sigma, X_prev, p_prev):
    p_i = []
    dX_prev = 1e-10  # Adjust the step size to a smaller value
    for j in range(2):
        integral_1 = 0
        integral_2 = 0
        # trace 3
        for X in np.arange(Lu + dX_prev, np.inf, dX_prev):
            integral_1 += (1 - L(Lu, rho, mu, sigma)) * f(X_prev) * dX_prev
        for X in np.arange(-np.inf, Lu, dX_prev):
            integral_2 += (1 - L(Lu, rho, mu, sigma)) * f(X_prev) * dX_prev
        p_i.append(p_prev[j] * integral_1 + (1 - p_prev[j]) * integral_2)
    return p_i

def compute_ARL(P):
    # Compute ARL (Average Run Length)
    return 1 / P

def obj(individual):
    p_prev = individual

    lambda_0 = p_prev[0]
    lambda_1 = p_prev[1]

    for i in range(2, n+1):
        # trace 2
        p_i = compute_p_i(Lu, rho, mu, sigma, X_prev, p_prev)
        lambda_0 += p_i[0]
        lambda_1 += p_i[1]
        p_prev = p_i

    P_0 = lambda_0
    P_1 = lambda_1
    ARL_0 = compute_ARL(P_0)
    ARL_1 = compute_ARL(P_1)

    return ARL_0, ARL_1

# DEAP initialization
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", obj)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

population_size = 10
num_generations = 5

population = toolbox.population(n=population_size)
hall_of_fame = tools.HallOfFame(maxsize=1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min, axis=0)

# Run the genetic algorithm
# trace 5
#population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations,
#                                          stats=stats, halloffame=hall_of_fame)

population = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations,
                                stats=stats, halloffame=hall_of_fame)

# Get the best individual from the hall of fame
best_individual = hall_of_fame[0]

# Evaluate the best individual
# trace 4
ARL_0, ARL_1 = obj(best_individual)

print("Optimized ARL_0:", ARL_0)
print("Optimized ARL_1:", ARL_1)