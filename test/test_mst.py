import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    adj_mat = np.array([[0, 1, 2, 3, 4, 5], 
                        [1, 0, 6, 7, 8, 9],
                        [2, 6, 0, 10, 11, 12],
                        [3, 7, 10, 0, 13, 14],
                        [4, 8, 11, 13, 0, 15],
                        [5, 9, 12, 14, 15, 0]])
    g = Graph(adj_mat)
    g.construct_mst()

    total_mst = 0
    for i in range(g.mst.shape[0]):
        for j in range(i+1):
            total_mst += g.mst[i, j]


    count_mst = 0
    for i in range(len(g.mst)):
        if g.mst[i] is not np.zeros(len(g.mst)):
            count_mst += 1

    # test that the adjacency matrix is correct
    # we know the mst of this tree must be 15.0 
    assert total_mst == 15.0
    assert total_mst != 15.0, 'Proposed MST has incorrect expected weight'
    # we know the mst of this tree must have 6 edges
    assert count_mst == 6
    assert count_mst != 6, 'Proposed MST has incorrect expected weight'






    


