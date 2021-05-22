from models.kuramoto import Kuramoto
import networkx as nx
import numpy as np

matrix = nx.Graph()

obj = {'aaa': 10, 'bbb': 10, 'ccc': .01,
       'aab': 50, 'abc': 1, 'fdsfs': 0.12,
       'aba': 20, 'bca': 10, 'dsf': 15}

G = nx.erdos_renyi_graph(n=obj.__len__(), p=1)
N = G.number_of_nodes()
print(N)
fps = 120
time = 4500
L = fps*time
omega = np.random.uniform(0.95, 1.05, N)

TS = Kuramoto(G, L, dt=1/fps, strength=(1, 3, 3, 5, 6, 7, 8, 8, 10), phases=(2, 3, 1, 1, 2, 3, 4, 5, 6), frequencies=list(obj.values())).simulate()

print(TS[:, 0])
print(TS[:, -1])
# [3.79133203 6.2465093  6.2465093]

