
"""
Generate an all-pairs road-distance matrix using Dijkstra.

Distances are calculated on the actual directed Delhi OSM road graph.
The matrix stores distances in metres.
"""

import csv
import sys
from pathlib import Path

import osmnx as ox

# Add the main project folder to Python's import path
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import build_adjacency, dijkstra

DATA_DIR = ROOT / "data"

LOCATIONS_FILE = DATA_DIR / "locations.csv"
MAPPING_FILE = DATA_DIR / "location_road_nodes.csv"
GRAPH_FILE = DATA_DIR / "delhi_north_drive.graphml"
OUTPUT_FILE = DATA_DIR / "distance_matrix_m.csv"


def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():

    print("=" * 75)
    print("ALL-PAIRS DELHI ROAD DISTANCE MATRIX")
    print("Algorithm: Dijkstra")
    print("=" * 75)

    # Read locations in dataset order.
    location_rows = read_csv(LOCATIONS_FILE)

    location_ids = [
        row["location_id"].strip().upper()
        for row in location_rows
    ]

    if len(set(location_ids)) != len(location_ids):
        raise ValueError("Duplicate location IDs found.")

    # Read road-node mapping.
    mapping_rows = read_csv(MAPPING_FILE)

    mapping = {}

    for row in mapping_rows:

        location_id = row["location_id"].strip().upper()

        if location_id in mapping:
            raise ValueError(f"Duplicate mapping for {location_id}")

        mapping[location_id] = int(row["road_node_id"])

    missing = set(location_ids) - set(mapping)

    if missing:
        raise ValueError(
            "Missing road-node mappings: " + ", ".join(sorted(missing))
        )

    # Load real Delhi road network.
    print("\nLoading OpenStreetMap road network...")

    graph = ox.io.load_graphml(GRAPH_FILE)

    adjacency = build_adjacency(graph)

    print(f"Road nodes: {graph.number_of_nodes():,}")
    print(f"Directed edges: {graph.number_of_edges():,}")

    # Verify all mapped nodes exist.
    for location_id in location_ids:

        node = mapping[location_id]

        if node not in adjacency:
            raise ValueError(
                f"Mapped node for {location_id} is missing from the graph."
            )

    # Calculate directed shortest distances.
    matrix = {}

    print("\nRunning Dijkstra from each location...")

    for source_id in location_ids:

        source_node = mapping[source_id]

        distances, _ = dijkstra(
            adjacency,
            source_node
        )

        matrix[source_id] = {}

        for target_id in location_ids:

            target_node = mapping[target_id]

            matrix[source_id][target_id] = distances.get(target_node)

        print(f"Completed: {source_id}")

    # Print the matrix in kilometres.
    print("\n" + "=" * 105)
    print("SHORTEST ROAD DISTANCE MATRIX (KM)")
    print("=" * 105)

    print(
        f"{'FROM / TO':<13}"
        + "".join(f"{location_id:>13}" for location_id in location_ids)
    )

    print("-" * 105)

    for source_id in location_ids:

        values = []

        for target_id in location_ids:

            distance = matrix[source_id][target_id]

            if distance is None:
                values.append(f"{'UNREACHABLE':>13}")
            else:
                values.append(f"{distance / 1000:>13.3f}")

        print(f"{source_id:<13}" + "".join(values))

    print("-" * 105)

    # Save the matrix in metres to avoid unnecessary rounding.
    fieldnames = ["from_location_id"] + [
        f"{location_id}_m" for location_id in location_ids
    ]

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for source_id in location_ids:

            row = {"from_location_id": source_id}

            for target_id in location_ids:

                distance = matrix[source_id][target_id]

                row[f"{target_id}_m"] = (
                    "" if distance is None else f"{distance:.3f}"
                )

            writer.writerow(row)

    # Count unreachable directed pairs.
    unreachable = [
        (source, target)
        for source in location_ids
        for target in location_ids
        if matrix[source][target] is None
    ]

    # Check directional differences.
    asymmetric_pairs = []

    for i, source_id in enumerate(location_ids):

        for target_id in location_ids[i + 1:]:

            forward = matrix[source_id][target_id]
            reverse = matrix[target_id][source_id]

            if (
                forward is not None
                and reverse is not None
                and abs(forward - reverse) > 1
            ):
                asymmetric_pairs.append(
                    (source_id, target_id, forward, reverse)
                )

    print("\nMATRIX GENERATION SUMMARY")
    print("-" * 75)
    print(f"Locations: {len(location_ids)}")
    print(f"Directed distance calculations: {len(location_ids) ** 2}")
    print(f"Unreachable directed pairs: {len(unreachable)}")
    print(f"Directionally asymmetric pairs: {len(asymmetric_pairs)}")
    print(f"Saved file: {OUTPUT_FILE}")

    if unreachable:
        print("\nWARNING: Some directed routes are unreachable.")

        for source, target in unreachable:
            print(f"{source} -> {target}")

    if asymmetric_pairs:
        print("\nExamples of directionally different distances:")

        for source, target, forward, reverse in asymmetric_pairs[:10]:

            print(
                f"{source} -> {target}: {forward / 1000:.3f} km | "
                f"{target} -> {source}: {reverse / 1000:.3f} km"
            )

    print("\nDistance matrix generation completed.")


if __name__ == "__main__":

    try:
        main()

    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)
