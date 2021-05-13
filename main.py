import numpy as np
from models.kuramoto import Kuramoto
from networkx import binomial_graph as to_binomial, to_numpy_array as to_array

matrix = to_array(to_binomial(n=5, p=1))
print(matrix.T)

model = Kuramoto(coupling=3, dt=0.01, total_time=100000, nodes_count=len(matrix), vibration_array=[1, 2, 3, 4, 5])
act_mat = model.run(connectivity_matrix=matrix)

times = [i for i in range(100000)]
for time in times:
    print(act_mat[:, time])