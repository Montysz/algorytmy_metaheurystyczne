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

def solution_print(graph, tour, path='here.png'):
    edgelist = []
    
    if p.edge_weight_format == "FULL_MATRIX":
        for i in range(len(tour) - 1):
            edgelist.append((tour[i]-1, tour[i + 1]-1))
        edgelist.append( (tour[len(tour) - 1]-1, tour[0]-1) )
    else:
        for i in range(len(tour) - 1):
            edgelist.append((tour[i], tour[i + 1]))
        edgelist.append( (tour[len(tour) - 1], tour[0]) )
    print(edgelist)
    n = nx.get_node_attributes(graph, 'coord')
    l = len(n)
    

    if l > 0 and n[1] != None:

        nx.draw(graph, nx.get_node_attributes(graph, 'coord'), edgelist = edgelist, with_labels=True, node_color = 'green')
    else:
        pos = nx.spring_layout(graph, seed=7, ) 
        nx.draw_networkx_nodes(graph, pos, node_size=300, node_color = 'green')
        nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=2)
    plt.savefig(path) 
    #plt.show()
    plt.close()    


#test_all()

test_random()


name = "st70"
p = read("tsp/"+str(name)+".tsp")
G = p.get_graph()

t = read("tsp/"+str(name)+".opt.tour")
opt = t.tours[0]



test_random()

'''

kr = kRandom(G, 100)[1]
print(evaluate(G,kr))

n = nearest_neighbour(G, 2)
print(evaluate(G,n))

ne = nearest_neighbour_extended(G)
print(evaluate(G,ne))

to = two_opt(G, kr.copy())
print(evaluate(G,to))

solution_print(G, kr, 'kr.png')
solution_print(G, n, 'n.png')
solution_print(G, ne, 'ne.png')
solution_print(G, to, 'to.png')
'''