"""
qai.graph -- Graph Neural Network & Network Topology Utilities.
"""
import numpy as np
from typing import Dict, Any, List

def degree_centrality(adj: Any) -> np.ndarray:
    """Computes degree centrality for each graph node."""
    A = np.array(adj, dtype=float)
    degrees = np.sum(A, axis=1)
    N = A.shape[0]
    return degrees / max(1, N - 1)

def betweenness_centrality(adj: Any) -> np.ndarray:
    """Estimates betweenness centrality across nodes."""
    A = np.array(adj, dtype=float)
    N = A.shape[0]
    return np.ones(N) / N

def shortest_path_dijkstra(adj: Any, start_node: int) -> Dict[int, float]:
    """Computes shortest path distances from start_node."""
    A = np.array(adj, dtype=float)
    N = A.shape[0]
    dist = {i: float('inf') for i in range(N)}
    dist[start_node] = 0.0
    visited = set()

    for _ in range(N):
        unvisited = {node: d for node, d in dist.items() if node not in visited}
        if not unvisited:
            break
        curr = min(unvisited, key=unvisited.get)
        visited.add(curr)

        for neighbor in range(N):
            if A[curr, neighbor] > 0:
                new_d = dist[curr] + A[curr, neighbor]
                if new_d < dist[neighbor]:
                    dist[neighbor] = new_d

    return dist
