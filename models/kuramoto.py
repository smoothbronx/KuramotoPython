from collections import defaultdict

import networkx as nx
import numpy as np
from functools import wraps
import scipy.integrate as it
import warnings
from typing import Any


def ensure_unweighted(graph):
    for _, _, attr in graph.edges(data=True):
        if not np.isclose(attr.get("weight", 1.0), 1.0):
            new_graph = graph.__class__()
            new_graph.add_nodes_from(graph)
            new_graph.add_edges_from(graph.edges)
            warnings.warn("Coercing weighted graph to unweighted.", RuntimeWarning)
            return new_graph
    return graph


class DataLengthException(Exception):
    def __init__(self, message: str):
        self.__message = message

    def raise_this(self):
        raise self

    def __repr__(self):
        return self.__message


def unweighted(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [ensure_unweighted(arg) if issubclass(arg.__class__, nx.Graph) else arg for arg in args]
        return func(*args, **kwargs)
    return wrapper


class Kuramoto:
    __results = defaultdict(dict)

    @unweighted
    def __init__(self, graph, time: int = 20, dt: float = 0.01, strength=None, phases=None, frequencies=None):
        self.graph = graph
        self.dt = dt
        self.graph_array = nx.to_numpy_array(self.graph)
        self.nodes_count = self.graph.number_of_nodes()
        self.time = np.linspace(self.dt, time * self.dt, time)
        self.array_of_ones = np.ones(self.nodes_count)

        self.strength = (np.array(strength) if strength.__len__() == self.nodes_count else
                         DataLengthException('Connectivity must be equal to the number of oscillators').raise_this()) \
            if strength is not None else np.random.uniform(0.1, 10, self.nodes_count)

        self.null_theta = (phases if phases.__len__() == self.nodes_count
                           else DataLengthException('Phases must be equal to the number of oscillators').raise_this()) \
            if phases is not None else 2 * np.pi * np.random.rand(self.nodes_count)

        self.omega = (frequencies if frequencies.__len__() == self.nodes_count
                      else DataLengthException('Frequencies must be equal to the number of oscillators').raise_this()) \
            if frequencies is not None else np.random.uniform(0.9, 1.1, self.nodes_count)

    def simulate(self):
        ts_wrapper = it.odeint(self.theta, self.null_theta, self.time, self.strength, self.omega, self.graph_array)
        ts = np.flipud(ts_wrapper.T) % (2 * np.pi)

        self.__results["internal_frequencies"] = self.omega
        self.__results["ground_truth"] = self.graph
        self.__results["TS"] = ts
        return ts

    def getResults(self):
        return self.__results

    def __set_res(self, path: str, obj: Any):
        return self.__results.__setitem__(path, obj)

    def theta(self, null_theta, time, strength, omega, graph_array):
        return omega + strength / self.nodes_count * (self.graph_array * np.sin(np.outer(self.array_of_ones, null_theta) - np.outer(null_theta, self.array_of_ones))).dot(self.array_of_ones)


# graph!, time added, dt later, strength later, phases=None, frequencies=None
# class KuramotoHandler:
#     response = defaultdict(dict)
#
#     def __init__(self, data: dict):
#         self.objects_list = data.get("objects")
#         self.fps = data.get("fps")
#         self.time = data.get("time")
#         self.objects_names = tuple(self.objects_list.keys())



