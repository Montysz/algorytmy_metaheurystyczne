import random
import time
from numpy import number
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import pylab
import cyth 



def read(path):
    with open(path) as f:
        text = f.read()
        problem = tsplib95.parse(text)
        
    return problem

def random_instance(n, seed, type, a = 2, b = 100):
    G = nx.Graph()
    seed = seed^(a+b)^n
    random.seed(seed)

    if type == "not_dirested":
        for i in range(1,n+1):
            for j in range(i + 1,n+1):
                w = random.randint(a,b)
                G.add_edge(i, j, weight = w)
 
                     
    return G

def distance_matrix(graph):
    return nx.to_numpy_matrix(graph, graph.nodes).getA()
    #for i in x:
    #    print(i)

def solution_print(graph, tour):
    edgelist = []
    
    for i in range(len(tour) - 1):
        edgelist.append((tour[i], tour[i + 1]))
    edgelist.append((tour[len(tour) - 1], tour[0]))

    n = nx.get_node_attributes(graph, 'coord')
    l = len(n)

    if l > 0 and n[1] != None:

        nx.draw(graph, nx.get_node_attributes(graph, 'coord'), edgelist = edgelist, with_labels=True, node_color = 'green')
    else:
        pos = nx.spring_layout(graph, seed=7, ) 
        nx.draw_networkx_nodes(graph, pos, node_size=300, node_color = 'green')
        nx.draw_networkx_edges(graph, pos, edgelist=edgelist, width=2)
    plt.savefig('here.png') 
    plt.show()

def graph_print(graph):
    n = nx.get_node_attributes(graph, 'coord')

    l = len(n)

    if l > 0 and n[1] != None:
        nx.draw(graph, nx.get_node_attributes(graph, 'coord'), with_labels=True, node_color = 'green')
    else:
        pos = nx.spring_layout(graph, seed=7) 
        nx.draw_networkx_nodes(graph, pos, node_size=300, node_color = 'green')
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for (u, v, d) in graph.edges(data=True)], width=0.5)
    
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

def tests():
    import os
    directory = 'tsp'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and "tour"  not in f:
            print(f)
            p = read(f)
            G = p.get_graph()
            graph_print(G)


''' 
tests()
name = "berlin52"
p = read("tsp/"+str(name)+".tsp")
G1 = p.get_graph()
#graph_print(G1)
t = read("tsp/"+str(name)+".opt.tour")
tour = t.tours[0]

#solution_print(G1, tour)
ref = evaluate(G1, tour)
prd(G1, tour, ref)

G2 = random_instance(52, 1414, "not_dirested")
graph_print(G2)
solution_print(G2, tour)
'''
name = "berlin52"
p = read("tsp/"+str(name)+".tsp")
G1 = p.get_graph()

t = read("tsp/"+str(name)+".opt.tour")
tour = t.tours[0]

def kRandom(k, G):
    Graph = distance_matrix(G)
    bestAns = 1e9
    bestTour= []
    for i in range(k):
        ans = np.random.permutation(range(1, len(Graph[0])+1))

        curAns = evaluate(G, ans)
        if(bestAns > curAns):
            bestTour = ans
            bestAns = curAns
    return bestAns, bestTour

'''
s = time.time()
kRandomAns, kRandomTour = kRandom(99999, G1)
print(kRandomAns, kRandomTour)
e = time.time()
print(e - s, "python")
ref = evaluate(G1, tour)
prd(G1, tour, kRandomAns)

s = time.time()
kRandomAns, kRandomTour = cyth.kRandom(99999, G1, distance_matrix(G1))
print(kRandomAns, kRandomTour)
e = time.time()
print(e - s, "cython")
ref = evaluate(G1, tour)
prd(G1, tour, kRandomAns)

solution_print(G1, kRandomTour)
'''



def notYetVisited(tab):
    ans = False
    for i in range(len(tab)):
        if(tab[i] == 0):
            ans = True
            break
    return ans
    
def nearestNeighbour(s, G):
    Graph = distance_matrix(G)
    visited = [0 for i in range(len(Graph[0])+1)]
    visited[s] = 1
    visited[0] = 1
    solution = []
    while(notYetVisited(visited)):
        curEdges =  [(Graph[s-1][i],i+1) for i in range(len(Graph[0]))]

        sortedEdges = sorted(curEdges, key=lambda k: k[0])
        
        for i in range(1, len(Graph[0])+1):
            if not visited[sortedEdges[i][1]]:
                visited[sortedEdges[i][1]] = True
                solution.append(sortedEdges[i][1])
                break
    return solution


s = time.time()
nearestNeighbourAns = nearestNeighbour(2, G1)
e = time.time() 
print(e-s)

ref = evaluate(G1, nearestNeighbourAns)
prd(G1, tour, ref)

s = time.time()
nearestNeighbourAns = cyth.nearestNeighbour(2, G1, distance_matrix(G1))
e = time.time() 
print(e-s)



ref = evaluate(G1, nearestNeighbourAns)
prd(G1, tour, ref)

#solution_print(G1, tour)
#solution_print(G1, nearestNeighbourAns)

