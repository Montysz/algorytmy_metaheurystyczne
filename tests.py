
import resource
import time
import random
import numpy as np
import os

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
        res.write(f"kRandom_lenght kRandom_time   nNeighbour_lenght nNeighbour_time   nNeighbour-ext_lenght nNeighbour-ext_time   two-opt_lenght two-opt_time   n\n")

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

def test_compare(path, dir_name,  to = False, n = False, ne = False, kr = False, opt_path = None, n_start = False, kr_k = False):

    
    dir_path = os.getcwd() + f"/tour_visual/{dir_name}"
    try: os.mkdir(dir_path)
    except OSError: print("fail")
    problem_name = f"{path[path.find('/')+1:path.find('.')]}"
    p = read(path)
    G = p.get_graph()
    alg = {}


    if n_start == False : n_start = random.randint(1,(G.number_of_nodes()))
    if kr_k == False : kr_k = 5000

    if opt_path: 
        opt_tour = read(opt_path).tours[0]
        solution_print(G, opt_tour.copy(),p, path=dir_path+"/"+problem_name+"_opt-sol.png")
        alg['opt'] = (evaluate(G, opt_tour))
        
    if to:
        #start_tour = (np.random.permutation( [*range(1,(G.number_of_nodes())),1])
        start_tour = []
        start, end = 1, G.number_of_nodes()
        start_tour.extend(range(start, end))
        start_tour.append(end)
        start_tour = np.random.permutation(start_tour)

        to = two_opt(G , start_tour.copy())
        solution_print(G, to.copy(),p, path=dir_path+"/"+problem_name+"_to-sol.png")
        solution_print(G, start_tour.copy(), p, path=dir_path+"/"+problem_name+"_to_start.png")
        alg['to']  = (evaluate(G, to))
    if n:
        n = nearest_neighbour(G , n_start)
        solution_print(G, n.copy(),p, path=dir_path+"/"+problem_name+"_n-sol.png")
        alg['n'] = (evaluate(G, n))
    if ne:
        ne = nearest_neighbour_extended(G)
        solution_print(G, ne.copy(),p, path=dir_path+"/"+problem_name+"_ne-sol.png")
        alg['ne'] = (evaluate(G, ne))
    if kr:
        kr = kRandom(G, kr_k)
        
        solution_print(G, kr[1].copy(),p, path=dir_path+"/"+problem_name+"_kr-sol.png")
        alg['kr'] = (evaluate(G, kr[1]))
    #print(alg)

    #plots part
    names = list(alg.keys())
    values = list(alg.values())
    plt.pyplot.clf()
    plt.pyplot.bar(range(len(alg)), values, tick_label=names)
    
    
    dir_path = os.getcwd() + f"/plots/{dir_name}"
    try: os.mkdir(dir_path)
    except OSError: print("fail")   
    
    plt.pyplot.savefig(f"{dir_path}/{problem_name}")
