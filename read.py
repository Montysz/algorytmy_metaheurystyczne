from cProfile import label
import random
from numpy import number
from sqlalchemy import true
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def read(path):
    with open(path) as f:
        text = f.read()
        problem = tsplib95.parse(text)
        
    return problem

def random_instance(n, seed, type, a = 2, b = 100):
    G = nx.Graph()
    seed = seed^(a+b)^n
    random.seed(seed)

    for i in range(1,n+1):
        for j in range(i + 1,n+1):
            w = random.randint(a,b)
            G.add_edge(i, j, weight = w)
                     
    pos = nx.spring_layout(G, seed=7) 
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color = 'green')
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for (u, v, d) in G.edges(data=True)], width=2)
    #plt.show()
    return G

def distance_matrix(graph):
    x = nx.to_numpy_matrix(graph, graph.nodes).getA()
    for i in x:
        print(i)

def solution_print(graph, tour):
    edgelist = []
    for i in range(len(tour) - 1):
        edgelist.append((tour[i], tour[i + 1]))
    edgelist.append((tour[len(tour) - 1], tour[0]))
    
    pos = nx.spring_layout(graph, seed=7 )
    #nx.draw_networkx_labels(graph, pos)
    #nx.draw_networkx_nodes(graph,  pos, node_size=300, node_color = 'green')
    #nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=2)

    city_list =np.array(edgelist)
    
    nx.draw_networkx(graph, edgelist = edgelist)
    #plt.scatter(city_list[:,0],city_list[:,1])
    plt.plot()
    plt.show()


def evaluate(graph, tour):
    edgelist = []
    for i in range(len(tour) - 1):
        edgelist.append((tour[i], tour[i + 1]))
    edgelist.append((tour[len(tour) - 1], tour[0]))
    sum = 0
    m = nx.to_numpy_matrix(graph, graph.nodes).getA()
    for i in edgelist:
        sum = sum + m[i[0] - 1][i[1] - 1]
    return sum


def prd(graph, x, ref):
    print(100 * ((evaluate(graph, x) - ref) / ref),"%")

name = "ulysses16"
p = read("tsp/"+str(name)+".tsp")
G1 = p.get_graph()
t = read("tsp/"+str(name)+".opt.tour")
tour = t.tours[0]

solution_print(G1, tour)

ref = evaluate(G1, tour)
prd(G1, tour, ref)



