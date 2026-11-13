
"""
Core algorithms for Disaster Relief Vehicle Routing.

Stage 1:
    Build adjacency list from the OpenStreetMap road graph.

Stage 2:
    Dijkstra's shortest-path algorithm.

All road distances are measured in metres.
"""

import heapq
from math import inf, isfinite


def build_adjacency(graph):
    """
    Convert a directed NetworkX MultiDiGraph into an adjacency list.

    Each edge weight is the OpenStreetMap road length in metres.
    Parallel edges are preserved.
    One-way road directions are preserved.
    """

    if not graph.is_directed():
        raise ValueError("The road graph must be directed.")

    adjacency = {node: [] for node in graph.nodes}

    for source, destination, attributes in graph.edges(data=True):

        if "length" not in attributes:
            raise ValueError(
                f"Road edge {source} -> {destination} has no length."
            )

        try:
            distance = float(attributes["length"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid road length on edge {source} -> {destination}."
            ) from error

        if not isfinite(distance) or distance < 0:
            raise ValueError(
                f"Invalid or negative road distance: {distance}"
            )

        adjacency[source].append((destination, distance))

    return adjacency


def dijkstra(adjacency, source, target=None):
    """
    Calculate shortest road distances using Dijkstra's algorithm.

    Parameters:
        adjacency: Weighted directed adjacency list.
        source: Starting road-network node.
        target: Optional destination node.

    Returns:
        distances: Shortest distances from source, in metres.
        predecessors: Previous node for path reconstruction.
    """

    if source not in adjacency:
        raise KeyError(f"Starting node {source} does not exist.")

    distances = {source: 0.0}
    predecessors = {}

    priority_queue = [(0.0, source)]

    while priority_queue:

        current_distance, current_node = heapq.heappop(
            priority_queue
        )

        # Ignore outdated entries in the priority queue.
        if current_distance > distances.get(current_node, inf):
            continue

        # Stop when the requested destination is reached.
        if target is not None and current_node == target:
            break

        for neighbour, road_length in adjacency[current_node]:

            new_distance = current_distance + road_length

            if new_distance < distances.get(neighbour, inf):

                distances[neighbour] = new_distance
                predecessors[neighbour] = current_node

                heapq.heappush(
                    priority_queue,
                    (new_distance, neighbour)
                )

    return distances, predecessors


def reconstruct_path(predecessors, source, target):
    """
    Reconstruct the shortest path from source to destination.

    Returns:
        Ordered list of road-network node IDs.
        Empty list if no path exists.
    """

    if source == target:
        return [source]

    if target not in predecessors:
        return []

    path = [target]
    current_node = target

    while current_node != source:

        current_node = predecessors.get(current_node)

        if current_node is None:
            return []

        path.append(current_node)

    path.reverse()

    return path
