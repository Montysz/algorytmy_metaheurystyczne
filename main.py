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



def read(path):
    with open(path) as f:
        text = f.read()
        problem = tsplib95.parse(text)
        
    return problem

def random_instance(n, seed, type, a = 2, b = 100):
    G = None
    seed = seed^(a+b)^n
    random.seed(seed)

    if type == "Symmetric":
        G = nx.Graph()
        for i in range(1,n+1):
            for j in range(i + 1,n+1):
                w = random.randint(a,b)
                G.add_edge(i, j, weight = w)
    
    if type == "EUC_2D":
        G = nx.Graph()
        s = """ NAME : ramdominstance
                COMMENT : .
                TYPE : TSP
                DIMENSION : 100
                EDGE_WEIGHT_TYPE : EUC_2D
                NODE_COORD_SECTION
                """
        for i in range(1, n+1):
            x = random.randint(a,b)
            y = random.randint(a,b)
            s = s + f"{i} {x} {y} \n"
        s = s + "EOF\n"
        problem = tsplib95.parse(s)   
        G = problem.get_graph() 
        
    if type == "Asymmetric":
        G = nx.DiGraph()
        for i in range(1,n+1):
            for j in range(1,n+1):
                w = random.randint(a,b)
                G.add_edge(i, j, weight = w)

    
                     
    return G

def distance_matrix(graph):
    return nx.to_numpy_matrix(graph, graph.nodes).getA()
    #for i in x:
    #    print(i)

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
    print(f"{100*((evaluate(graph,x)-ref)/ref)}%")

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



def kRandom(G, k):
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


def notYetVisited(tab):
    ans = False
    for i in range(len(tab)):
        if(tab[i] == 0):
            ans = True
            break
    return ans
    
def nearest_neighbour(G, s):
    Graph = distance_matrix(G)
    visited = [0 for i in range(len(Graph[0])+1)]
    visited[s] = 1
    visited[0] = 1
    solution = []
    solution.append(s)
    while(notYetVisited(visited)):
        curEdges =  [(Graph[s-1][i],i+1) for i in range(len(Graph[0]))]
        sortedEdges = sorted(curEdges, key=lambda k: k[0])
        
        for i in range(0, len(Graph[0])):
            if not visited[sortedEdges[i][1]]:
                visited[sortedEdges[i][1]] = True
                solution.append(sortedEdges[i][1])
                break
    return solution

def nearest_neighbour_extended(G):
    Graph = distance_matrix(G)
    best_sol = 1e9
    best_ans = None
    for s in range(1, len(Graph[0]) + 1):
        solution = nearest_neighbour(G, s)
        cur_sol = evaluate(G, solution)
        if cur_sol < best_sol:
            best_sol = cur_sol
            best_ans = solution
    return best_ans

def cost(cost_mat, route):
    
    
    ans = cost_mat[np.roll(route, 1), route].sum()
    
    return ans

def two_opt(G, route):
    best = route
    for i in range(len(best)):
        best[i] = best[i] - 1
    Graph = distance_matrix(G)   
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)):
            for j in range(i+2, len(route)):
                new_route = route.copy()
                new_route[i:j] = route[j-1:i-1:-1] # this is the 2woptSwap
                if cost(Graph, new_route) < cost(Graph, best):  # what should cost be?
                    best = new_route
                    improved = True
        route = best
    for i in range(len(best)):
        best[i] = best[i] + 1
    return best


name = "st70"
p = read("tsp/"+str(name)+".tsp")
G = p.get_graph()

t = read("tsp/"+str(name)+".opt.tour")
opt = t.tours[0]



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
#    random_instance(n, seed, type, a = 2, b = 100):
#    Symmetric
#    EUC_2D
#    Asymmetric

G  = random_instance(32, 2138, "Symmetric")


kr = kRandom(G, 100)[1]
print(evaluate(G, kr))

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
