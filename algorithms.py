from problem  import *
import numpy as np
import time
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
                new_route[i:j] = route[j-1:i-1:-1]
                if cost(Graph, new_route) < cost(Graph, best):
                    best = new_route
                    improved = True
        route = best
    for i in range(len(best)):
        best[i] = best[i] + 1
    return best

def two_opt_time(G, route, t):
    best = route
    for i in range(len(best)):
        best[i] = best[i] - 1
    Graph = distance_matrix(G)
    start = time.time() 
    while True:
        for i in range(1, len(route)):
            for j in range(i+2, len(route)):
                new_route = route.copy()
                new_route[i:j] = route[j-1:i-1:-1]
                if cost(Graph, new_route) < cost(Graph, best):
                    best = new_route
        route = best
        if time.time() - start > t:
            break 
    for i in range(len(best)):
        best[i] = best[i] + 1
    return best

def kRandom_time(G, t):
    Graph = distance_matrix(G)
    bestAns = 1e9
    bestTour= []
    start = time.time() 
    while True:
        ans = np.random.permutation(range(1, len(Graph[0])+1))

        curAns = evaluate(G, ans)
        if(bestAns > curAns):
            bestTour = ans
            bestAns = curAns
        if time.time() - start > t:
            break 
    return bestAns, bestTour
