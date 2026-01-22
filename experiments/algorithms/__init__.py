# Comparison Algorithms
"""
Comparison Algorithms Module
============================

This module contains implementations of standard pathfinding algorithms
for comparison with AILS.

Algorithms:
- A* (with various heuristics)
- Dijkstra's Algorithm
- Breadth-First Search (BFS)
- Jump Point Search (JPS)
- Bidirectional A*
"""

from .astar import AStar
from .dijkstra import Dijkstra
from .bfs import BFS
from .jps import JumpPointSearch
from .bidirectional import BidirectionalAStar

__all__ = [
    'AStar',
    'Dijkstra',
    'BFS',
    'JumpPointSearch',
    'BidirectionalAStar',
]
