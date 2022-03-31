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
import resource
from algorithms import *
from tests import *
from problem import *

  

#test_all()
#test_random_time(type ="Symmetric",a = 2, b = 150, c = 5, k = 10)
#test_random_time(type ="Asymmetric",a = 2, b = 150, c = 5, k = 10)
#test_random_time(type ="EUC_2D",a = 2, b = 150, c = 5, k = 10)

#test_random(type ="Symmetric",a = 2, b = 100, c = 5, k = 10)
#test_random(type ="Asymmetric",a = 2, b = 100, c = 5, k = 10)
#test_random(type ="EUC_2D",a = 2, b = 100, c = 5, k = 10)
#test_random()



name = "st70"
p = read("tsp/"+str(name)+".tsp")
G = p.get_graph()

t = read("tsp/"+str(name)+".opt.tour")
opt = t.tours[0]

test_compare(
    path = "atsp/br17.atsp", 
    dir_name = 'br17',  

    to = True,
    n = True,
    ne = True,
    kr = True,
)

