import random
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt
import numpy
from cpython cimport array
cimport numpy
from libcpp cimport bool

cpdef int test2(int x):
    cdef int y = 0
    cdef int i
    for i in range(x):
        y += i
    return y

cpdef int evaluate(graph, tour,m):
    edgelist = []
    cdef int j 
    for j in range(len(tour) - 1):
        edgelist.append((tour[j], tour[j + 1]))
    edgelist.append((tour[len(tour) - 1], tour[0]))
    cdef int sum = 0
    for i in edgelist:
        sum = sum + m[i[0] - 1][i[1] - 1]
    return sum


cpdef kRandom(int k, G, Graph):
    cdef float  bestAns = 10e9

    cdef numpy.ndarray ans
    cdef numpy.ndarray bestTour
    cdef int i 
    for i in range(k):
        ans = numpy.random.permutation(range(1,len(Graph[0])+1))
        curAns = evaluate(G, ans,Graph)
        if(bestAns > curAns):
            print(curAns)
            bestTour = ans
            bestAns = curAns
    return bestAns, bestTour


cpdef notYetVisited(list tab):
    cpdef bint ans = False
    cdef int i
    cdef int l = len(tab) 
    for i in range(l):
        if(tab[i] == 0):
            ans = True
            break
    return ans

cdef nearestNeighbour(int s, G, Graph):
    cdef list visited 
    visited = [0 for i in range(len(Graph[0])+1)]
    visited[s] = 1
    visited[0] = 1
    
    cdef list solution = []
    solution.append(s)
    cpdef bint yet_visited = notYetVisited(visited)
    while(notYetVisited(visited)):
        curEdges =  [(Graph[s-1][i],i+1) for i in range(len(Graph[0]))]

        sortedEdges = sorted(curEdges, key=lambda k: k[0])
        
        for i in range(1, len(Graph[0])+1):
            if not visited[sortedEdges[i][1]]:
                visited[sortedEdges[i][1]] = True
                solution.append(sortedEdges[i][1])
                break
    return solution


'''
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
                new_route[i:j] = route[j-1:i-1:-1] 
                if cost(Graph, new_route) < cost(Graph, best): 
                    best = new_route
                    improved = True
        route = best
    for i in range(len(best)):
        best[i] = best[i] + 1
    return best
'''
