import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):

        self.mst = None

        # Prism's algorithm
        n = len(self.adj_mat)
        visited = set()        
        start = 0
        end = 0
        heap = [(0, start, end)]
        mst = np.zeros((self.adj_mat.shape))
        # loop through 
        while len(visited) < n:
            # 2. Initialize a set S to be empty
            # 3. While Q is not empty:
            #   a. Remove the vertex u with the smallest key from Q
            #   b. Add u to S
            #   c. For each vertex v adjacent to u:
            #       i. If v is in Q and the edge weight between u and v is less than v's key:
            #           1. Update v's key to be the edge weight between u and v
            #           2. Update v's parent to be u
            weight, start, end = heapq.heappop(heap)
            if end not in visited:
                visited.add(end)
                mst[start][end] = weight
                mst[end][start] = weight
                for i in range(n):
                    if i not in visited and self.adj_mat[end][i] != 0:
                        heapq.heappush(heap, (self.adj_mat[end][i], end, i))

        self.mst = mst
        