
from typing import List
import random
import numpy
import math
from pickle import TRUE
import random
import time
from numpy import number
#from sqlalchemy import false
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


"""name = "pr76"
p = read("tsp/"+str(name)+".tsp")
#p = read("atsp/"+str(name)+".atsp")

G = p.get_graph()
dm = distance_matrix(G)



TOTAL_CHROMOSOME = len(dm[0]) 

POPULATION_SIZE = len(dm[0]) * 20
MAX_GENERATION = 100
MUTATION_RATE = 0.3
"""

def NN(A, start):
    path = [start]
    cost = 0
    N = A.shape[0]
    
    mask = np.ones(N, dtype=bool)   
                                  
    mask[start] = False

    for _ in range(N-1):
        last = path[-1]
        next_ind = np.argmin(A[last][mask])
        next_loc = np.arange(N)[mask][next_ind] 
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

def create_genome(dm) -> Genome:

    genome = Genome()
    chance = random.random()
    #if chance > 0.9:
    #    genome.chromosome = random.sample(range(0, TOTAL_CHROMOSOME-1), TOTAL_CHROMOSOME-1)
    #else:
    genome.chromosome = NN(dm, random.randint(0, len(dm) - 1))

    genome.fitness = eval_chromosome(genome.chromosome, dm)


    return genome

def get_fittest_genome(genomes: List[Genome]) -> Genome:
    genome_fitness = [genome.fitness for genome in genomes]
    return genomes[genome_fitness.index(min(genome_fitness))]

def eval_chromosome(chromosome: List[int], dm) -> float:

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

def order_crossover(parents: List[Genome], TOTAL_CHROMOSOME, dm) -> Genome:
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
    child.fitness = eval_chromosome(child.chromosome, dm)

    #for i in range(len(child.chromosome) - 1):
    #    if i not in child.chromosome:
    #        return order_crossover(parents)

    return child

def scramble_mutation(genome: Genome, TOTAL_CHROMOSOME, G, dm) -> Genome:
    chance = random.random()
    if chance > 0.02:
        subset_length = random.randint(2, 6)
        start_point = random.randint(0, TOTAL_CHROMOSOME - subset_length)
        subset_index = [start_point, start_point + subset_length]

        subset = genome.chromosome[subset_index[0]:subset_index[1]]
        random.shuffle(subset)

        genome.chromosome[subset_index[0]:subset_index[1]] = subset
    elif chance > 0.01:
        genome.chromosome = two_opt_iter(G, genome.chromosome.copy(), 1)

    else:
        genome.chromosome = two_opt2(G, genome.chromosome.copy())

    genome.fitness = eval_chromosome(genome.chromosome, dm)
    return genome

def reproduction(population: List[Genome], TOTAL_CHROMOSOME, MUTATION_RATE, G, dm) -> Genome:
    parents = [tournament_selection(population, len(population) // 10), random.choice(population)] 

    child = order_crossover(parents, TOTAL_CHROMOSOME, dm)
    
    if random.random() < MUTATION_RATE:
        scramble_mutation(child, TOTAL_CHROMOSOME, G, dm)

    return child


def genSetup(path = "tsp/berlin52.tsp", G = None, maxGen = 100, mutationRate = 0.3, popSize = 0):
    
    
    return main(path, G, maxGen, mutationRate, popSize)

def genTime(path = "tsp/berlin52.tsp", G=None, maxTime = 100, mutationRate = 0.3,  popSize = 0):
    start = time.time() 
    generation = 0
    if(G == None):
        p = read(path)

        G = p.get_graph()
    else:
        G = G

    dm = distance_matrix(G)

    TOTAL_CHROMOSOME = len(dm[0]) 

    POPULATION_SIZE = len(dm[0]) * 20 if popSize == 0 else popSize
    MUTATION_RATE = mutationRate
    MAX_TIME = maxTime

    population = [create_genome(dm) for _ in range (POPULATION_SIZE)]

    all_fittest = []
    all_pop_size = []
    last_best = float('inf') 
    no_progress = 1
    while time.time() - start < MAX_TIME:
        generation += 1
        best = get_fittest_genome(population).fitness
        if(best <  last_best):
            print("Generation: {0} -- Population size: {1} -- Best Fitness: {2}"
                .format(generation, len(population), get_fittest_genome(population).fitness))
        else:
            print("Generation: {0} -- Population size: {1} -- Best Fitness: {2}, no progress {3}"
                .format(generation, len(population), get_fittest_genome(population).fitness, no_progress))
        if(best < last_best): 
            last_best = best
            no_progress = 1
        else:
            no_progress += 1
        
        childs = []
        for x in range(int(POPULATION_SIZE * 0.2)):
            child = reproduction(population, TOTAL_CHROMOSOME, MUTATION_RATE, G, dm)
            childs.append(child)
        population.extend(childs)


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
            print("Stagnation")
            pop = population.copy()
            pop.sort(key = lambda x: x.fitness, reverse=False)
            
            population = pop[0:len(pop) // 2]

        if no_progress % 273 == 0:
            print("run Two-opt")
            genome = Genome()
            genome.chromosome = two_opt2(G, get_fittest_genome(population).chromosome.copy())
            genome.fitness = eval_chromosome(genome.chromosome, dm)

            print("Val", genome.fitness)
            population.append(genome)
            genome = Genome()
            genome.chromosome = two_opt2(G,population[random.randint(0,len(population)-1)].chromosome.copy())
            genome.fitness = eval_chromosome(genome.chromosome, dm)


        if no_progress % 250 == 0: 
            print("Kor loss")
            genome = Genome()
            genome.chromosome = get_fittest_genome(population).chromosome
            genome.fitness = get_fittest_genome(population).fitness
            population = [genome]

            for _ in range (POPULATION_SIZE):
                population.append(create_genome(dm))

        all_fittest.append(get_fittest_genome(population))

        all_pop_size.append(len(population))
    print(get_fittest_genome(population).fitness)
    for i in range(len(get_fittest_genome(population).chromosome)):
        if i not in get_fittest_genome(population).chromosome:
            print(i ,"error")
    tmp = get_fittest_genome(population).chromosome.copy()
    for i, _ in enumerate(tmp):
        tmp[i] += 1
    print(tmp, "od 1")
    return tmp
    print(len(get_fittest_genome(population).chromosome))
    print(get_fittest_genome(population).chromosome, "od 0")




def main(path, G=None, maxGen = 100, mutationRate = 0.3, popSize = 0):
    generation = 0
    if(G == None):
        p = read(path)

        G = p.get_graph()
    else:
        G = G
    dm = distance_matrix(G)

    TOTAL_CHROMOSOME = len(dm[0]) 

    POPULATION_SIZE = len(dm[0]) * 20 if popSize == 0 else popSize
    MAX_GENERATION = maxGen
    MUTATION_RATE = mutationRate

    population = [create_genome(dm) for _ in range (POPULATION_SIZE)]

    all_fittest = []
    all_pop_size = []
    last_best = float('inf') 
    no_progress = 1
    while generation != MAX_GENERATION:
        generation += 1
        best = get_fittest_genome(population).fitness
        if(best <  last_best):
            print("Generation: {0} -- Population size: {1} -- Best Fitness: {2}"
                .format(generation, len(population), get_fittest_genome(population).fitness))
        else:
            print("Generation: {0} -- Population size: {1} -- Best Fitness: {2}, no progress {3}"
                .format(generation, len(population), get_fittest_genome(population).fitness, no_progress))
        if(best < last_best): 
            last_best = best
            no_progress = 1
        else:
            no_progress += 1
        
        childs = []
        for x in range(int(POPULATION_SIZE * 0.2)):
            child = reproduction(population, TOTAL_CHROMOSOME, MUTATION_RATE, G, dm)
            childs.append(child)
        population.extend(childs)


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
                if(len(population)< len(dm)):
                    population.append(create_genome(dm))
        if no_progress % 25 == 0:  
            print("Stagnation")
            pop = population.copy()
            pop.sort(key = lambda x: x.fitness, reverse=False)
            
            population = pop[0:len(pop) // 2]
            while(len(population)< len(dm)):
                population.append(create_genome(dm))

        if no_progress % 273 == 0:
            print("run Two-opt")
            genome = Genome()
            genome.chromosome = two_opt2(G, get_fittest_genome(population).chromosome.copy())
            genome.fitness = eval_chromosome(genome.chromosome, dm)

            print("Val", genome.fitness)
            population.append(genome)
            genome = Genome()
            genome.chromosome = two_opt2(G,population[random.randint(0,len(population)-1)].chromosome.copy())
            genome.fitness = eval_chromosome(genome.chromosome, dm)
            while(len(population)< len(dm)):
                population.append(create_genome(dm))


        if no_progress % 250 == 0: 
            print("Kor loss")
            genome = Genome()
            genome.chromosome = get_fittest_genome(population).chromosome
            genome.fitness = get_fittest_genome(population).fitness
            population = [genome]

            while(len(population)< len(dm)):
                population.append(create_genome(dm))
            

        all_fittest.append(get_fittest_genome(population))

        all_pop_size.append(len(population))
    print(get_fittest_genome(population).fitness)
    for i in range(len(get_fittest_genome(population).chromosome)):
        if i not in get_fittest_genome(population).chromosome:
            print(i ,"error")
    tmp = get_fittest_genome(population).chromosome.copy()
    for i, _ in enumerate(tmp):
        tmp[i] += 1
    print(tmp, "od 1")
    return tmp
    print(len(get_fittest_genome(population).chromosome))
    print(get_fittest_genome(population).chromosome, "od 0")

if __name__ == "__main__":
    genTime(popSize=0, maxTime=10)
    #genSetup()
    

ref = 108159


#our = [20, 10, 4, 15, 18, 17, 14, 22, 11, 19, 25, 7, 23, 27, 16, 24, 8, 1, 28, 6, 12, 9, 5, 26, 0, 3, 2, 21, 13]
#our = [115, 69, 7, 53, 89, 95, 110, 23, 59, 15, 60, 0, 75, 28, 119, 29, 31, 91, 27, 44, 77, 80, 93, 85, 13, 86, 74, 43, 45, 49, 97, 41, 16, 48, 117, 19, 106, 68, 64, 42, 107, 24, 18, 116, 30, 65, 21, 84, 17, 112, 67, 90, 57, 99, 32, 51, 78, 12, 50, 10, 114, 1, 81, 2, 79, 26, 4, 62, 72, 56, 82, 66, 36, 61, 98, 9, 34, 103, 105, 113, 76, 63, 92, 20, 108, 87, 96, 11, 94, 38, 52, 8, 22, 102, 118, 3, 37, 6, 40, 55, 73, 14, 58, 104, 71, 39, 47, 101, 100, 109, 111, 35, 83, 5, 88, 54, 46, 70, 25, 33]
#our = get_fittest_genome(population).chromosome
#our = [26, 35, 38, 39, 37, 63, 66, 64, 68, 65, 69, 67, 48, 43, 49, 47, 55, 51, 50, 18, 30, 15, 7, 13, 8, 11, 10, 9, 12, 3, 60, 59, 57, 44, 46, 45, 58, 27, 22, 25, 24, 6, 0, 1, 5, 4, 23, 52, 54, 53, 20, 17, 16, 19, 41, 36, 42, 40, 32, 29, 28, 56, 33, 31, 62, 61, 34, 14, 2, 21]
#prd(G, our, ref)
'''
[39, 38, 36, 37, 18, 17, 16, 15, 74, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 75, 76, 1, 23, 22, 21, 25, 24, 46, 45, 44, 48, 47, 69, 68, 70, 67, 50, 49, 51, 66, 65, 71, 72, 73, 64, 63, 62, 61, 58, 57, 56, 55, 52, 53, 54, 42, 43, 28, 27, 26, 20, 19, 31, 30, 29, 32, 33, 35, 34, 40, 41, 60, 59] od 1
76
'''
#print(p.comment)
#print(eval_chromosome(our), "od 0")
#print(f"{100*((eval_chromosome(our)-ref)/ref)}%")
#best solution(by us): [0, 27, 5, 11, 8, 4, 25, 28, 2, 1, 20, 19, 9, 3, 14, 17, 16, 13, 21, 10, 18, 24, 6, 22, 26, 15, 12, 23, 7]
#[20, 10, 4, 15, 18, 17, 14, 22, 11, 19, 25, 7, 23, 27, 16, 24, 8, 1, 28, 6, 12, 9, 5, 26, 0, 3, 2, 21, 13]