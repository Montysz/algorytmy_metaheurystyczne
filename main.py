from pickle import TRUE
import random
import time
from numpy import number
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

#test_random(type ="Symmetric",a = 10, b = 50, c = 5, k = 10)
#test_random(type ="Asymmetric",a = 10, b = 50, c = 5, k = 10)
#test_random(type ="EUC_2D",a = 10, b = 50, c = 5, k = 10)


#test_all()

#test_random_time(type ="Symmetric",a = 10, b = 50, c = 5, k = 10)
#test_random_time(type ="Asymmetric",a = 10, b = 50, c = 5, k = 10)
#test_random_time(type ="EUC_2D",a = 10, b = 50, c = 5, k = 10)

#test_random(type ="Symmetric",a = 2, b = 100, c = 5, k = 10)
#test_random(type ="Asymmetric",a = 2, b = 100, c = 5, k = 10)
#test_random(type ="EUC_2D",a = 2, b = 100, c = 5, k = 10)
#test_random()

name = "ft70"
#p = read("tsp/"+str(name)+".tsp")
p = read("atsp/"+str(name)+".atsp")

G = p.get_graph()

'''
name = "70"
p = read("tsp/"+str(name)+".tsp")
G = p.get_graph()

t = read("tsp/"+str(name)+".opt.tour")
opt = t.tours[0]


test_compare(
    path = "tsp/att48.tsp", 
    dir_name = 'att48',  
    opt_path = "tsp/att48.opt.tour",
    to = True,
    n = True,
    ne = True,
    kr = True,
)
'''
def main():

    if sys.argv[1] == "all":
        test_all()
    elif sys.argv[1] == "random":
        arg_dict = {}
        if "-a" in  sys.argv: arg_dict['a'] = int(sys.argv[sys.argv.index("-a") + 1])
        if "-b" in  sys.argv: arg_dict['b'] = int(sys.argv[sys.argv.index("-b") + 1])
        if "-c" in  sys.argv: arg_dict['c'] = int(sys.argv[sys.argv.index("-c") + 1])
        if "-k" in  sys.argv: arg_dict['k'] = int(sys.argv[sys.argv.index("-k") + 1])
        if "-type" in  sys.argv: arg_dict['type'] = (sys.argv[sys.argv.index("-type") + 1])
        print(arg_dict)
        test_random(**arg_dict)

    elif sys.argv[1] == "compare":
        arg_dict = {}
        if "-path" in  sys.argv: arg_dict['path'] = (sys.argv[sys.argv.index("-path") + 1])
        if "-dir" in  sys.argv: arg_dict['dir_name'] = (sys.argv[sys.argv.index("-dir") + 1])
        if "-opt" in  sys.argv: arg_dict['opt_path'] = (sys.argv[sys.argv.index("-opt") + 1])
        if "-opt_val" in  sys.argv: arg_dict['opt_val'] = (sys.argv[sys.argv.index("-opt_val") + 1])

        if "-to" in  sys.argv: arg_dict['to'] = True
        if "-n" in  sys.argv: arg_dict['n'] = True
        if "-ne" in  sys.argv: arg_dict['ne'] = True
        if "-kr" in  sys.argv: arg_dict['kr'] = True
        if "-tb" in  sys.argv: arg_dict['tb'] = True
        if "-gen" in  sys.argv: arg_dict['gen'] = True

        if "-n_start" in  sys.argv: arg_dict['n_start'] = int(sys.argv[sys.argv.index("-n_start") + 1])
        if "-kr_k" in  sys.argv: arg_dict['kr_k'] = int(sys.argv[sys.argv.index("-kr_k") + 1])
        if "-tb_size" in  sys.argv: arg_dict['tb_size'] = int(sys.argv[sys.argv.index("-tb_size") + 1])
        if "-tb_iter" in  sys.argv: arg_dict['tb_iter'] = int(sys.argv[sys.argv.index("-tb_iter") + 1])

        if "-gen_iter" in  sys.argv: arg_dict['gen_iter'] = int(sys.argv[sys.argv.index("-gen_iter") + 1])
        if "-gen_mut" in  sys.argv: arg_dict['gen_mut'] = int(sys.argv[sys.argv.index("-gen_mut") + 1])
        if "-gen_pop" in  sys.argv: arg_dict['gen_pop'] = int(sys.argv[sys.argv.index("-gen_pop") + 1])
        if "-gen_time" in  sys.argv: arg_dict['gen_time'] = int(sys.argv[sys.argv.index("-gen_time") + 1])

        print(arg_dict)
        test_compare(**arg_dict)
    

#python3 main.py compare -path tsp/rat99.tsp -dir rat99 -opt_val 1211 -to -tb -gen -gen_iter 100 tb_iter 50
#python3 main.py random -a 35 -b 40 -c 1 -k 10 -type Symmetric
#python3 main.py compare -path tsp/att48.tsp -dir att48 -opt_val 10628 -to -ne -tb


#test_all()

#test_random_time(type ="Asymmetric",a = 2, b = 150, c = 5, k = 10)
#test_random_time(type ="EUC_2D",a = 2, b = 150, c = 5, k = 10)


#test_random(type ="Asymmetric",a = 2, b = 100, c = 5, k = 10)
#test_random(type ="EUC_2D",a = 2, b = 100, c = 5, k = 10)
#test_random()
if __name__ == "__main__":
    name = "ft70"
    p = read("atsp/"+str(name)+".atsp")
    G = p.get_graph()
    #print(distance_matrix(G))
    main()
