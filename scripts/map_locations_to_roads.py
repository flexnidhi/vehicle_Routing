
"""Map real Delhi study locations to their nearest OSM road nodes."""

import csv
from pathlib import Path

import osmnx as ox

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

COORDINATES_FILE = DATA_DIR / "location_coordinates.csv"
ROAD_GRAPH_FILE = DATA_DIR / "delhi_north_drive.graphml"
OUTPUT_FILE = DATA_DIR / "location_road_nodes.csv"


def read_coordinates():
    if not COORDINATES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {COORDINATES_FILE}")

    with COORDINATES_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No locations found in coordinate CSV.")

    return rows


def map_locations():
    print("=" * 70)
    print("DELHI LOCATION TO ROAD-NODE MAPPING")
    print("=" * 70)

    if not ROAD_GRAPH_FILE.exists():
        raise FileNotFoundError(
            "Road graph not found. Download it before running this script."
        )

    print("\nLoading Delhi road network...")
    graph = ox.io.load_graphml(ROAD_GRAPH_FILE)

    print(f"Road nodes loaded: {graph.number_of_nodes():,}")

    locations = read_coordinates()
    output_rows = []

    print("\nFinding nearest road nodes...")
    print("-" * 90)

    for location in locations:
        location_id = location["location_id"].strip()
        name = location["location_name"].strip()

        latitude = float(location["latitude"])
        longitude = float(location["longitude"])

        # OSMnx uses X = longitude and Y = latitude.
        nearest_node = ox.distance.nearest_nodes(
            graph,
            X=longitude,
            Y=latitude
        )

        node_data = graph.nodes[nearest_node]

        road_latitude = float(node_data["y"])
        road_longitude = float(node_data["x"])

        offset_m = ox.distance.great_circle(
            latitude,
            longitude,
            road_latitude,
            road_longitude
        )

        output_rows.append({
            "location_id": location_id,
            "location_name": name,
            "latitude": latitude,
            "longitude": longitude,
            "road_node_id": nearest_node,
            "road_latitude": road_latitude,
            "road_longitude": road_longitude,
            "snap_offset_m": round(float(offset_m), 2)
        })

        print(
            f"{location_id:<7} {name[:43]:<45} "
            f"Node: {nearest_node}  Offset: {offset_m:.1f} m"
        )

    # Save the mapping for future algorithm calculations.
    fieldnames = [
        "location_id",
        "location_name",
        "latitude",
        "longitude",
        "road_node_id",
        "road_latitude",
        "road_longitude",
        "snap_offset_m"
    ]

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("-" * 90)
    print(f"Total locations mapped: {len(output_rows)}")
    print(f"Mapping saved to: {OUTPUT_FILE}")

    large_offsets = [
        row for row in output_rows
        if row["snap_offset_m"] > 100
    ]

    if large_offsets:
        print("\nWARNING: Check these locations because their")
        print("nearest road nodes are more than 100 metres away:")
        for row in large_offsets:
            print(
                f"{row['location_id']}: "
                f"{row['snap_offset_m']} metres"
            )
    else:
        print("\nAll locations are within 100 metres of their nearest road node.")

    print("\nLocation mapping completed.")


if __name__ == "__main__":
    try:
        map_locations()
    except (FileNotFoundError, ValueError, KeyError, OSError) as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)
