
import resource
import time
import random
import numpy as np

from algorithms import *
from problem import *



def test_all():
    import os
    directory = 'tsp'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and "tour"  not in f:
            print(f)
            p = read(f)
            G = p.get_graph()
            kr = kRandom(G, 1000)[1]
            print("kRandom")
            n = nearest_neighbour(G, 2)
            print("nearest_neighbour")
            ne = nearest_neighbour_extended(G)
            print("nearest_neighbour_extended")
            to = two_opt(G, kr.copy())
            print("two-opt")
            with open("results/test_all.txt", "a") as res:
                print(evaluate(G,ne))
                res.write(f"{evaluate(G,kr)}    ")
                res.write(f"{evaluate(G,n)}    ")
                res.write(f"{evaluate(G,ne)}    ")
                res.write(f"{evaluate(G,to)}    ")
                res.write(f"{filename}          \n")

def test_random(type ="Symmetric",a = 3, b = 100, c = 1, k = 10):
    kr_time = 0
    kr_space = 0
    kr_res = 0
    n_time = 0
    n_space = 0
    n_res = 0
    ne_time = 0
    ne_space = 0
    ne_res = 0
    to_time = 0
    to_space = 0
    to_res = 0  
    with open(f"results/test_random_{type}_{a}-{b}.txt", "w") as res:
        res.write(f"kRandom_lenght kRandom_time   nNeighbour_lenght nNeighbour_time   nNeighbour-ext_lenght nNeighbour-ext_time   two_opt_lenght two_opt-time   n\n")

        for i in range(a, b+1, c):
            print(f"{i}/{b}")
            for j in range(k):
                seed = random.randint(0,1000000)

                G = random_instance(i, seed, type, 2, 100)
                
                time_start = time.perf_counter()
                space_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                kr = kRandom(G, 1000)[1]
                time_end = time.perf_counter()
                space_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                kr_time += time_end - time_start
                kr_space += space_end - space_start
                kr_res += evaluate(G,kr)
                #print("kRandom")
                
                time_start = time.perf_counter()
                space_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                n = nearest_neighbour(G, 2)
                time_end = time.perf_counter()
                space_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                n_time += time_end - time_start
                n_space += space_end - space_start
                n_res += evaluate(G,n)
                #print("nearest_neighbour")

                time_start = time.perf_counter()
                space_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0        
                ne = nearest_neighbour_extended(G)
                time_end = time.perf_counter()
                space_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                ne_time += time_end - time_start
                ne_space += space_end - space_start
                ne_res += evaluate(G,ne)
                #print("nearest_neighbour_extended")
                
                time_start = time.perf_counter()
                space_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                to = two_opt(G, np.random.permutation(kr.copy()))
                time_end = time.perf_counter()
                space_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                to_time += time_end - time_start
                to_space += space_end - space_start
                to_res += evaluate(G,to)
                #print("two_opt")

            res.write(f"{kr_res / k} {kr_time / k}   ")
            res.write(f"{n_res / k} {n_time / k}    ")
            res.write(f"{ne_res / k} {ne_time / k}    ")
            res.write(f"{to_res / k} {to_time / k}    ")
            res.write(f"{i}          \n") 


def test_random_time(type ="Symmetric",a = 3, b = 100, c = 1, k = 10):
    kr_res = 0
    ne_res = 0
    ne_time = 0
    to_res = 0  
    with open(f"results/test_random_time_{type}_{a}-{b}.txt", "w") as res:
        res.write(f"kRandom_lenght        nNeighbour-ext_lenght    two-opt_lenght   n\n")

        for i in range(a, b+1, c):
            print(f"{i}/{b}")
            for j in range(k):
                seed = random.randint(0,1000000)

                G = random_instance(i, seed, type, 2, 100)
                
                time_start = time.time()
                ne = nearest_neighbour_extended(G)
                time_end = time.time()
                
                ne_time = time_end - time_start
                #print(ne_time)
                ne_res += evaluate(G,ne)
                #print("nearest_neighbour_extended")


                kr = kRandom_time(G, ne_time)[1]
                kr_res += evaluate(G,kr)
                #print("kRandom")
                to = two_opt_time(G, np.random.permutation(kr.copy()), ne_time)
                to_res += evaluate(G,to)
                #print("two_opt")

            res.write(f"{kr_res / k}    ")
            res.write(f"{ne_res / k}     ")
            res.write(f"{to_res / k}     ")
            res.write(f"{i}          \n") 