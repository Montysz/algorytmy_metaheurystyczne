import time
import random
import numpy as np
import os
import resource
from sympy import false
from genetic import *
from algorithms import *
from problem import *

from tabu import *


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
    tb_time = 0
    tb_space = 0
    tb_res = 0  
    gen_time = 0
    gen_space = 0
    gen_res = 0
    with open(f"results/test_random_{type}_{a}-{b}.txt", "w") as res:
        #res.write(f"kRandom_lenght kRandom_time   nNeighbour_lenght nNeighbour_time   nNeighbour-ext_lenght nNeighbour-ext_time   two-opt_lenght two-opt_time   tabu_lenght tabu_time   n\n")
        res.write(f"nNeighbour-ext_lenght nNeighbour-ext_time   two-opt_lenght two-opt_time   tabu_lenght tabu_time   gen_lenght gen_time n\n")

        for i in range(a, b+1, c):
            print(f"{i}/{b}")
            
            for j in range(k):
                seed = random.randint(0,1000000)

                G = random_instance(i, seed, type, 2, 100)
                
                """time_start = time.perf_counter()
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
                #print("nearest_neighbour")"""

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
                to = two_opt(G, np.random.permutation(ne.copy()))
                time_end = time.perf_counter()
                space_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                to_time += time_end - time_start
                to_space += space_end - space_start
                to_res += evaluate(G,to)
                #print("two_opt")


                time_start = time.perf_counter()
                space_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                tb = tabuSetup(iterations=int(i/10), G = G)
                time_end = time.perf_counter()
                space_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                tb_time += time_end - time_start
                tb_space += space_end - space_start
                tb_res += evaluate(G,tb)



                time_start = time.perf_counter()
                space_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                gen = genSetup(G = G, maxGen=int(i), mutationRate=0.3, popSize=0)
                time_end = time.perf_counter()
                space_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
                gen_time += time_end - time_start
                gen_space += space_end - space_start
                gen_res += evaluate(G,gen)

            #res.write(f"{kr_res / k} {kr_time / k}   ")
            #res.write(f"{n_res / k} {n_time / k}    ")
            res.write(f"{ne_res / k} {ne_time / k}    ")
            res.write(f"{to_res / k} {to_time / k}    ")
            res.write(f"{tb_res / k} {tb_time / k}    ")
            res.write(f"{gen_res / k} {gen_time / k}    ")
            res.write(f"{i}          \n") 
            ne_res = 0
            to_res = 0
            tb_res = 0
            ne_time = 0
            to_time = 0
            tb_time = 0
            gen_time = 0
            gen_space = 0
            gen_res = 0


def test_random_time(type ="Symmetric",a = 3, b = 100, c = 1, k = 10):
    kr_res = 0
    ne_res = 0
    ne_time = 0
    to_res = 0  
    tb_res = 0
    tb_time = 0
    tb_space = 0
    gen_res = 0
    with open(f"results/test_random_time_{type}_{a}-{b}.txt", "w") as res:
        res.write(f"nNeighbour-ext_lenght two-opt_lenght   tabu_lenght gen_lenght  n\n")

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

                to = two_opt_time(G, np.random.permutation(ne.copy()), ne_time*5)
                to_res += evaluate(G,to)

                tb = tabuSetup2(time=ne_time, G = G)
                tb_res += evaluate(G,tb)
                #print("two_opt")
                gen = genTime(G=G, maxTime=ne_time, mutationRate=0.3, popSize=0)
                gen_res += evaluate(G, gen)

            
            res.write(f"{ne_res / k}     ")
            res.write(f"{to_res / k}     ")
            res.write(f"{tb_res / k}    ")
            res.write(f"{gen_res / k}    ")
            res.write(f"{i}          \n")
            ne_res = 0
            to_res = 0
            tb_res = 0
            gen_res = 0

def test_compare(path, dir_name, to = False, n = False, ne = False, kr = False, tb = False, opt_val = None ,opt_path = None, n_start = False, kr_k = False, tb_size = 7, tb_iter = 100, gen = False, gen_iter = 100, gen_mut = 0.3, gen_pop = 0, gen_time = 0):

    
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
    if opt_val:
        alg['opt'] = int(opt_val)
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
        print(alg['to'])
        if alg['opt']:
            print("to:", prdVal(alg['to'], alg['opt']))

    if n:
        n = nearest_neighbour(G , n_start)
        solution_print(G, n.copy(),p, path=dir_path+"/"+problem_name+"_n-sol.png")
        alg['n'] = (evaluate(G, n))
        if alg['opt']:
            print("n:", prdVal(alg['n'], alg['opt']))
    if ne:
        ne = nearest_neighbour_extended(G)
        solution_print(G, ne.copy(),p, path=dir_path+"/"+problem_name+"_ne-sol.png")
        alg['ne'] = (evaluate(G, ne))
        if alg['opt']:
            print("ne:", prdVal(alg['ne'], alg['opt']))
    if kr:
        kr = kRandom(G, kr_k)
        
        solution_print(G, kr[1].copy(),p, path=dir_path+"/"+problem_name+"_kr-sol.png")
        alg['kr'] = (evaluate(G, kr[1]))
        if alg['opt']:
            print("kr:", prdVal(alg['kr'], alg['opt']))
            
    if tb:
        tb = tabuSetup(path=path, iterations = tb_iter, size = tb_size)

        solution_print(G, tb.copy(),p, path=dir_path+"/"+problem_name+"_tb-sol.png")
        alg['tb'] = (evaluate(G, tb))
        if alg['opt']:
            print("tabu:", prdVal(alg['tb'], alg['opt']))
    if gen:
        gen = genSetup(path=path, maxGen=gen_iter, mutationRate=gen_mut, popSize=gen_pop)
        solution_print(G, gen.copy(), p, path=dir_path+"/"+problem_name+"_gen-sol.png")
        alg['gen'] = (evaluate(G, gen))
        if alg['opt']:
            print("Genetic:", prdVal(alg['gen'], alg['opt']))
    print(alg)

    #plots part
    names = list(alg.keys())
    values = list(alg.values())
    plt.pyplot.clf()
    plt.pyplot.bar(range(len(alg)), values, tick_label=names)
    
    
    dir_path = os.getcwd() + f"/plots/{dir_name}"
    try: os.mkdir(dir_path)
    except OSError: print("fail")   
    
    plt.pyplot.savefig(f"{dir_path}/{problem_name}")

def tabuListTest():

    
    
    path = "tsp/ulysses22.tsp"
    p = read(path)
    G = p.get_graph()
    alg = {}
    res = []
    index = []
    for i in range(1, 51, 1):
        tb = tabuSetup(iterations=200, size=i, G=G)
        print(evaluate(G, tb), " ", i)
        res.append(evaluate(G, tb))
        index.append(i)

    #plots parts
    names = list(index)
    values = list(res)
 
    dir_path = os.getcwd() + f"/plots/tabuList"
    try: os.mkdir(dir_path)
    except OSError: print("fail")   
    
   
    

    plt.pyplot.plot(names, values)
    plt.pyplot.suptitle("ulysses22")
    plt.pyplot.savefig(f"{dir_path}/tabuList")          
    plt.pyplot.clf() 

