#
# USI - Universita della svizzera italiana
#
# Machine Learning - Fall 2017
#
# Assignment 5: Evolutionary Computation
#
# (!) This code skeleton is just a recommendation.
# (!) Please describe all the solutions in your report
#

import numpy as np
import random
import heapq
from math import exp
import matplotlib.pyplot as plt
import seaborn


def get_fitness(chromosome):
    cnt = 0
    one = False
    new_pair = True
    for i in chromosome:
        if i == 0:
            one = False
            new_pair = True
        if i == 1:
            if new_pair & one:
                cnt += 1
                new_pair = False
            one = True
    return cnt


def discrete_sample(probabilities):
    index = 0
    length = len(probabilities)
    indices = np.arange(length)
    index = np.random.choice(indices, 1, probabilities)
    return index


def fitness_proportional_selection(fitnesses):
    probabilities = np.zeros_like(fitnesses, dtype=np.float)
    sum = np.sum(fitnesses)
    for i in range(0, len(fitnesses)):
        # print(fitnesses[i])
        probabilities[i] = float(fitnesses[i]) / float(sum)
    indices = np.arange(len(fitnesses))
    # print(indices)
    # print(probabilities)
    index = np.random.choice(indices, 1, p=probabilities)
    # print(index)
    return index[0]


def bitflip_mutatation(chromosome, mutation_rate):
    for i in chromosome:
        if random.random() < mutation_rate:
            if chromosome[i] == 0:
                chromosome[i] == 1
            else:
                chromosome[i] == 0
    pass


def one_point_crossover(parentA, parentB):
    index = np.random.randint(len(parentA))

    child1 = np.concatenate((parentA[0:index - 1], parentB[index:]))
    child2 = np.concatenate((parentB[0:index - 1], parentA[index:]))

    return child1, child2


def two_point_crossover(parentA, parentB):
    index1 = np.random.randint(1, high=len(parentA))
    index2 = np.random.randint(1, high=len(parentA))
    while index1 == index2 or abs(index2 - index2) == 1:
        # print(len(parentA))
        index2 = np.random.randint(len(parentA))
    if index1 < index2:
        child1_temp = np.concatenate(
            (parentA[0:(index1)], parentB[(index1):(index2)]))
        child1 = np.concatenate((child1_temp, parentA[index2:]))
        child2_temp = np.concatenate((
            parentB[0:(index1)], parentA[(index1):(index2)]))
        child2 = np.concatenate((child2_temp, parentB[index2:]))
    else:
        child1_temp = np.concatenate(
            (parentA[0:(index2)], parentB[(index2):(index1)]))
        child1 = np.concatenate((child1_temp, parentA[index1:]))
        child2_temp = np.concatenate((
            parentB[0:(index2)], parentA[(index2):(index1)]))
        child2 = np.concatenate((child2_temp, parentB[index1:]))

    return child1, child2


def generate_initial_population(length, population_size):
    population = [np.random.randint(2, size=length)
                  for i in range(0, population_size)]
    return population


def ga(length, population_size, mutation_rate, cross_over_rate=1.0, max_gen=1000):
    population = generate_initial_population(length, population_size)
    runstat = []
    for g in range(0, max_gen):
        fitnesses = [get_fitness(chromosome) for chromosome in population]
        runstat.append(fitnesses)
        mating_pool = [fitness_proportional_selection(
            fitnesses) for i in range(0, population_size)]
        new_population = []
        for i in range(0, population_size - 1, 2):
            parentA = population[mating_pool[i]]
            parentB = population[mating_pool[i + 1]]
            childA, childB = two_point_crossover(parentA, parentB)
            bitflip_mutatation(childA, mutation_rate)
            bitflip_mutatation(childB, mutation_rate)
            new_population.append(childA)
            new_population.append(childB)
        population = new_population
    return runstat, population


def plot_minmax_curve(run_stats):
    min_length = min(len(r) for r in run_stats)
    truncated_stats = np.array([r[:min_length] for r in run_stats])
    # print(truncated_stats)

    X = np.arange(truncated_stats.shape[1])
    means = truncated_stats.mean(axis=0)
    mins = truncated_stats.min(axis=0)
    maxs = truncated_stats.max(axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(means, '-o')

    ax.fill_between(X, mins[:, 0], maxs[:, 0], linewidth=0,
                    facecolor="b", alpha=0.3, interpolate=True)
    ax.fill_between(X, mins[:, 1], maxs[:, 1], linewidth=0,
                    facecolor="g", alpha=0.3, interpolate=True)
    ax.fill_between(X, mins[:, 2], maxs[:, 2], linewidth=0,
                    facecolor="r", alpha=0.3, interpolate=True)
    plt.show()


def run_part1(length=50):
    np.random.seed(106)
    run_stats = []
    for run in range(10):
        run_stat, _ = ga(length=length, population_size=200,
                         mutation_rate=0.001, max_gen=1000)
        # print(run_stat)
        run_stats.append(run_stat)
    plot_minmax_curve(run_stats)


def plot_part1(max_gen=10, population_size=50, run_length=10):
    np.random.seed(106)
    run_stats = []
    lengths = [5, 10, 20, 50]
    for length in lengths:
        avg_stats = []
        avg = np.zeros(max_gen)
        for run in range(run_length):
            run_stat, _ = ga(length=length, population_size=population_size,
                             mutation_rate=0.1, max_gen=max_gen)
            avg_stats.append(run_stat)
        for gen in range(max_gen):
            for run in range(run_length):
                # print(np.amax(avg_stats[run][gen]))
                avg[gen] += np.amax(avg_stats[run][gen])
        for gen in range(max_gen):
            avg[gen] = avg[gen] / run_length
        plt.plot(avg)
    plt.show()
    return avg


# part 2


def rosenbrock(x):
    return np.sum((1 - x[:-1])**2 + 100 * (x[1:] - x[:-1]**2)**2, axis=0)


def plot_surface():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    G = np.meshgrid(np.arange(-1.0, 1.5, 0.05), np.arange(-1.0, 1.5, 0.05))
    R = rosenbrock(np.array(G))

    fig = plt.figure(figsize=(14, 9))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(G[0], G[1], R.T, rstride=1,
                           cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_zlim(0.0, 500.0)
    ax.view_init(elev=50., azim=230)

    plt.show()


def sample_offspring(parents, lambda_, tau, tau_prime, epsilon0=0.001):
    beta = np.random.normal(0, tau_prime)
    child_pool = []
    for p in parents:
        for j in range(int(lambda_ / len(parents))):
            child = []
            sigma_prime = []
            beta_prime = []
            for i in range(int(len(p) / 2)):
                beta_prime.append(np.random.normal(0, tau))
            for i in range(int(len(p) / 2)):
                sigma_prime.append(p[int(len(p) / 2) + i]
                                   * exp(beta + beta_prime[i]))
            for i in range(int(len(p) / 2)):
                child.append(p[i] + np.random.normal(0, sigma_prime[i]))
            for i in range(int(len(p) / 2)):
                child.append(sigma_prime[i])
            child_pool.append(child)
    fitnesses = [rosenbrock(np.array(c)) for c in child_pool]
    return fitnesses, child_pool


def ES(N=5, mu=2, lambda_=100, generations=100, epsilon0=0.001):
    tau_prime = (1 / ((2 * N)**(1 / 2)))
    tau = (1 / ((2 * (N**(1 / 2))**(1 / 2))))
    parents = []
    fitnesses_plotting = []
    for i in range(0, mu):
        parents.append([np.random.randint(-5, 10, 1) for i in range(N)])
        for j in range(N):
            parents[i].append(0.1)

    for i in range(generations):
        new_parents = []
        fitnesses, children = sample_offspring(
            parents, lambda_, tau, tau_prime)
        fitnesses_plotting.append(np.amin(fitnesses))
        for i in range(mu):
            new_parents.append(children[np.argmin(fitnesses)])
            fitnesses.remove(np.amin(fitnesses))
            children.pop(np.argmin(fitnesses))
        parents = new_parents
    return fitnesses_plotting, parents[0]


def plot_ES_curve(F):
    min_length = min(len(f) for f in F)
    F_plot = np.array([f[:min_length] for f in F])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.mean(F_plot.T, axis=1))
    ax.fill_between(range(min_length), np.min(F_plot.T, axis=1), np.max(
        F_plot.T, axis=1), linewidth=0, facecolor="b", alpha=0.3, interpolate=True)
    ax.set_yscale('log')
    plt.show()


def run_part2(length=5):
    run_stats = []
    for i in range(10):
        fit, solution = ES(N=length, mu=10, lambda_=100,
                           epsilon0=0.0001, generations=500)
        run_stats.append(fit)
    plot_ES_curve(run_stats)


# plot_part1(max_gen=100, population_size=100, run_length=10)
# run_part1()
run_part2(length=50)
