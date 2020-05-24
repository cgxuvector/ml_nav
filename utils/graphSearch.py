"""
    This script contains the classic graph searching method:
        - Breadth-First Search
        - Dijkstra's Algorithm
        - Gready Best-First Search
        - A^* Search

    Map representation:
        - Grid map
"""
import numpy as np


class Graph(object):
    pass


if __name__ == '__main__':
    grid_map = np.load("./map.npy")
    print(grid_map)
    print(grid_map.shape)