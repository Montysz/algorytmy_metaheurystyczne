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
import resource
from algorithms import *
from tests import *
from problem import *
import os
import copy


def generateNeighbours(distanceMatrix):
    dict = {}
    
    for i in range(len(distanceMatrix)):
        for j in range(i+1, len(distanceMatrix[i])):
            cur = [i+1, j+1 , distanceMatrix[i][j]]
            
            if cur[0] not in dict:
                dict[cur[0]] = [[cur[1], cur[2]]]
            else:
                dict[cur[0]].append([cur[1], cur[2]])
            if cur[1] not in dict:
                dict[cur[1]] = [[cur[0], cur[2]]]
            else:
                dict[cur[1]].append([cur[0], cur[2]])

    return dict


def generateFirstSolution(dict):
    start = 1
    solution = []
    visiting = start
    dist = 0

    while visiting not in solution:
        minim = 10000
        for k in dict[visiting]:
            if int(k[1]) < int(minim) and k[0] not in solution:
                minim = k[1]
                bestNode = k[0]

        solution.append(visiting)
        dist = dist + int(minim)
        visiting = bestNode

    solution.append(start)
    position = 0

    for k in dict[solution[-2]]:
        if k[0] == start:
            break
        position += 1
    
    dist += int(dict[solution[-2]][position][1])-10000
    
    return solution, dist


def findNeighborhood(solution, dict):
    neighborhood = []

    for n in solution[1:-1]:
        idx1 = solution.index(n)
        for kn in solution[1:-1]:
            idx2 = solution.index(kn)
            if n == kn:
                continue

            tmp = copy.deepcopy(solution)
            tmp[idx1] = kn
            tmp[idx2] = n

            distance = 0

            for k in tmp[:-1]:
                nextNode = tmp[tmp.index(k) + 1]
                for i in dict[k]:
                    if i[0] == nextNode:
                        distance = distance + int(i[1])
            tmp.append(distance)

            if tmp not in neighborhood:
                neighborhood.append(tmp)

    lastIndex = len(neighborhood[0]) - 1

    neighborhood.sort(key=lambda x: x[lastIndex])
    return neighborhood


def tabuSearch(firstSolution, dist, dict, iters, size):
    count = 1
    solution = firstSolution
    tabuList = list()
    bestCost = dist
    bestSolutionEver = solution

    while count <= iters:
        neighborhood = findNeighborhood(solution, dict)
        bestSolutionIndex = 0
        bestSolution = neighborhood[bestSolutionIndex]
        bestCostIndex = len(bestSolution) - 1

        found = False
        while not found:
            i = 0
            while i < len(bestSolution):

                if bestSolution[i] != solution[i]:
                    firstExchangeNode = bestSolution[i]
                    secondExchangeNode = solution[i]
                    break
                i = i + 1

            if [firstExchangeNode, secondExchangeNode] not in tabuList and [secondExchangeNode, firstExchangeNode] not in tabuList:
                tabuList.append([firstExchangeNode, secondExchangeNode])
                found = True
                solution = bestSolution[:-1]
                cost = neighborhood[bestSolutionIndex][bestCostIndex]
                if cost < bestCost:
                    bestCost = cost
                    bestSolutionEver = solution
            else:
                bestSolutionIndex = bestSolutionIndex + 1
                bestSolution = neighborhood[bestSolutionIndex]

        if len(tabuList) >= size:
            tabuList.pop(0)

        count = count + 1

    return bestSolutionEver, bestCost


if __name__ == "__main__":
    
    #name = "st70"
    #name = "gr48"
    name = "bays29"

    p = read("tsp/"+str(name)+".tsp")
    G = p.get_graph()

    t = read("tsp/"+str(name)+".opt.tour")
    opt = t.tours[0]

    dict = generateNeighbours(distance_matrix(G))

    firstSolution, dist = generateFirstSolution(dict)
    iterations = 66
    size = 7
    
    kr = kRandom(G, 1000)[1]
    print("kRandom")
    print(evaluate(G, kr))
    n = nearest_neighbour(G, 1)
    print("nearest_neighbour")
    print(evaluate(G, n))
    ne = nearest_neighbour_extended(G)
    print("nearest_neighbour_extended")
    print(evaluate(G, ne))
    to = two_opt(G, ne.copy())
    print("two opt")
    print(evaluate(G, firstSolution[:-1]))
    

    
    bestSol, bestCost = tabuSearch(firstSolution, dist, dict, iterations, size)
    
    print(f"Optymalna\n{evaluate(G, opt)}")
    print(f"Tabu\n{bestCost}")
    prd(G, bestSol, evaluate(G, opt))
