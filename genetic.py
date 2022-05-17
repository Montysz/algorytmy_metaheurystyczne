from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from typing import List
import random
import numpy
import math
from pickle import TRUE
import random
import time
from numpy import number
from sqlalchemy import false
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import pylab
#import cyth
from algorithms import *
from tests import *
from problem import *
import os
import copy

name = "gr666"
p = read("tsp/"+str(name)+".tsp")
G = p.get_graph()
dm = distance_matrix(G)
# Zeroth index is start and end point

TOTAL_CHROMOSOME = len(dm[0]) 

POPULATION_SIZE = len(dm[0]) * 20
MAX_GENERATION = 10000
MUTATION_RATE = 0.3

def NN(A, start):
    path = [start]
    cost = 0
    N = A.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                                   # locations have not been visited
    mask[start] = False

    for _ in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += A[last, next_loc]


    return path

class Genome():
    def __init__(self):
        self.chromosome = []
        self.fitness = 0

    def __str__(self):
        return "Chromosome: {0} Fitness: {1}\n".format(self.chromosome, self.fitness) 
    
    def __repr__(self):
        return str(self)

def create_genome() -> Genome:

    genome = Genome()
    #genome.chromosome = random.sample(range(1, TOTAL_CHROMOSOME + 1), TOTAL_CHROMOSOME)
    genome.chromosome = NN(dm, random.randint(0, len(dm) - 1))
    genome.fitness = eval_chromosome(genome.chromosome)

    return genome

def get_fittest_genome(genomes: List[Genome]) -> Genome:
    genome_fitness = [genome.fitness for genome in genomes]
    return genomes[genome_fitness.index(min(genome_fitness))]

def eval_chromosome(chromosome: List[int]) -> float:

    fitness = 0
    for i in range(len(chromosome) - 1):
        fitness += dm[chromosome[i]][chromosome[i+1]]
    fitness += dm[chromosome[-1]][chromosome[0]]
    #fitness = evaluate(G, chromosome)
    return fitness

def tournament_selection(population:List[Genome], k:int) -> List[Genome]:
    selected_genomes = random.sample(population, k)
    selected_parent = get_fittest_genome(selected_genomes)
    return selected_parent

def order_crossover(parents: List[Genome]) -> Genome:
    parent1 = copy.deepcopy((parents[0].chromosome))
    parent2 = copy.deepcopy((parents[1].chromosome))

    #num = len(parent1)
    #child1 = parent1.copy()
    #child2 = parent2.copy()


    child_chro = [-1] * TOTAL_CHROMOSOME

    subset_length = random.randrange(2, 5)
    crossover_point = random.randrange(0, TOTAL_CHROMOSOME - subset_length)

    child_chro[crossover_point:crossover_point+subset_length] = parent1[crossover_point:crossover_point+subset_length]

    j, k = crossover_point + subset_length, crossover_point + subset_length
    while -1 in child_chro:
        if parent2[k] not in child_chro:
            child_chro[j] = parent2[k]
            j = j+1 if (j != TOTAL_CHROMOSOME-1) else 0
        k = k+1 if (k != TOTAL_CHROMOSOME-1) else 0

    child = Genome()
    child.chromosome = child_chro
    child.fitness = eval_chromosome(child.chromosome)

    #for i in range(len(child.chromosome) - 1):
    #    if i not in child.chromosome:
    #        return order_crossover(parents)

    return child

def scramble_mutation(genome: Genome) -> Genome:
    subset_length = random.randint(2, 6)
    start_point = random.randint(0, TOTAL_CHROMOSOME - subset_length)
    subset_index = [start_point, start_point + subset_length]

    subset = genome.chromosome[subset_index[0]:subset_index[1]]
    random.shuffle(subset)

    genome.chromosome[subset_index[0]:subset_index[1]] = subset
    genome.fitness = eval_chromosome(genome.chromosome)
    return genome

def reproduction(population: List[Genome]) -> Genome:
    parents = [tournament_selection(population, 20), random.choice(population)] 

    child = order_crossover(parents)
    
    if random.random() < MUTATION_RATE:
        scramble_mutation(child)

    return child

if __name__ == "__main__":
    generation = 0

    population = [create_genome() for _ in range (POPULATION_SIZE)]

    all_fittest = []
    all_pop_size = []
    last_best = float('inf') 
    no_progress = 1
    while generation != MAX_GENERATION:
        generation += 1
        print("Generation: {0} -- Population size: {1} -- Best Fitness: {2}"
            .format(generation, len(population), get_fittest_genome(population).fitness))
        
        if(get_fittest_genome(population).fitness < 2022): 
            print(get_fittest_genome(population).fitness)
            print(len(get_fittest_genome(population).chromosome))
            print(get_fittest_genome(population).chromosome)

            for i in range(len(get_fittest_genome(population).chromosome)):
                if i not in get_fittest_genome(population).chromosome:
                    print(i ,"kurwas")
            exit()
        
        childs = []
        for x in range(int(POPULATION_SIZE * 0.2)):
            child = reproduction(population)
            childs.append(child)
        population.extend(childs)


        best = get_fittest_genome(population).fitness
        if(best < last_best): 
            last_best = best
            no_progress = 1
        else:
            no_progress += 1

        # Kill weakness genome
        count = 0
        for genome in population:
            cur = (genome.fitness / best) - 1

            #c = random.random() * (cur - 1)
            r = random.random() /(2 ** (no_progress // 100 ))
            if no_progress % 25 == 0:
                r = 0
            #if (best / genome.fitness) > 0.95:
                #c = 1
            #if genome.fitness >  get_fittest_genome(population).fitness * 1.1:
            #if c < 0.5:
            if cur > r:
                population.remove(genome)
        if no_progress % 25 == 0:  
            print("Corwin")
            population = population[0:len(population) // 2]

        if no_progress % 273 == 0:
            print("run Two-opt")
            genome = Genome()
            genome.chromosome = two_opt(G, get_fittest_genome(population).chromosome.copy())
            genome.fitness = evaluate(G, genome.chromosome)

            print("Val", genome.fitness)
            population.append(genome)
            genome = Genome()
            genome.chromosome = two_opt(G,population[random.randint(0,len(population)-1)].chromosome.copy())
            genome.fitness = evaluate(G, genome.chromosome)


        if no_progress % 250 == 0: 
            print("Kor win")
            genome = Genome()
            genome.chromosome = get_fittest_genome(population).chromosome
            genome.fitness = get_fittest_genome(population).fitness
            population = [genome]

            for _ in range (POPULATION_SIZE):
                population.append(create_genome())

        all_fittest.append(get_fittest_genome(population))

        all_pop_size.append(len(population))
    print(get_fittest_genome(population).fitness)
    print(len(get_fittest_genome(population).chromosome))
    print(get_fittest_genome(population).chromosome)
    for i in range(len(get_fittest_genome(population).chromosome)):
        if i not in get_fittest_genome(population).chromosome:
            print(i ,"error")


our = [20, 10, 4, 15, 18, 17, 14, 22, 11, 19, 25, 7, 23, 27, 16, 24, 8, 1, 28, 6, 12, 9, 5, 26, 0, 3, 2, 21, 13]
our = [115, 69, 7, 53, 89, 95, 110, 23, 59, 15, 60, 0, 75, 28, 119, 29, 31, 91, 27, 44, 77, 80, 93, 85, 13, 86, 74, 43, 45, 49, 97, 41, 16, 48, 117, 19, 106, 68, 64, 42, 107, 24, 18, 116, 30, 65, 21, 84, 17, 112, 67, 90, 57, 99, 32, 51, 78, 12, 50, 10, 114, 1, 81, 2, 79, 26, 4, 62, 72, 56, 82, 66, 36, 61, 98, 9, 34, 103, 105, 113, 76, 63, 92, 20, 108, 87, 96, 11, 94, 38, 52, 8, 22, 102, 118, 3, 37, 6, 40, 55, 73, 14, 58, 104, 71, 39, 47, 101, 100, 109, 111, 35, 83, 5, 88, 54, 46, 70, 25, 33]
our = get_fittest_genome(population).chromosome

print(p.comment)
print(eval_chromosome(our))
ref = 7542
print(f"{100*((eval_chromosome(our)-ref)/ref)}%")
#best solution(by us): [0, 27, 5, 11, 8, 4, 25, 28, 2, 1, 20, 19, 9, 3, 14, 17, 16, 13, 21, 10, 18, 24, 6, 22, 26, 15, 12, 23, 7]
#[20, 10, 4, 15, 18, 17, 14, 22, 11, 19, 25, 7, 23, 27, 16, 24, 8, 1, 28, 6, 12, 9, 5, 26, 0, 3, 2, 21, 13]