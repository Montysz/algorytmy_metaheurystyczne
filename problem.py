import  tsplib95
import random
import networkx as nx
import matplotlib as plt


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
