
"""
Download the real Delhi driving road network from OpenStreetMap.

Input:
    data/location_coordinates.csv

Output:
    data/delhi_north_drive.graphml
"""

import csv
from pathlib import Path

import osmnx as ox
# Alternative global OpenStreetMap Overpass server
ox.settings.overpass_url = "https://maps.mail.ru/osm/tools/overpass/api"

# Allow a little more time for the request
ox.settings.requests_timeout = 240
# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

COORDINATES_FILE = DATA_DIR / "location_coordinates.csv"
ROAD_GRAPH_FILE = DATA_DIR / "delhi_north_drive.graphml"

# Small geographic buffer around all selected locations.
BUFFER = 0.005


def load_coordinates():
    """Read and validate the seven location coordinates."""

    if not COORDINATES_FILE.exists():
        raise FileNotFoundError(
            f"Coordinate file not found: {COORDINATES_FILE}"
        )

    with COORDINATES_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:

        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The coordinate file is empty.")

    coordinates = []

    for row in rows:

        location_id = row["location_id"].strip()

        latitude = float(row["latitude"])
        longitude = float(row["longitude"])

        if not -90 <= latitude <= 90:
            raise ValueError(f"Invalid latitude for {location_id}")

        if not -180 <= longitude <= 180:
            raise ValueError(f"Invalid longitude for {location_id}")

        coordinates.append({
            "id": location_id,
            "latitude": latitude,
            "longitude": longitude
        })

    print(f"Successfully loaded {len(coordinates)} locations.")

    return coordinates


def download_road_network(coordinates):
    """Download the driving network around all selected locations."""

    latitudes = [item["latitude"] for item in coordinates]
    longitudes = [item["longitude"] for item in coordinates]

    # Calculate bounding box with a small buffer.
    north = max(latitudes) + BUFFER
    south = min(latitudes) - BUFFER
    east = max(longitudes) + BUFFER
    west = min(longitudes) - BUFFER

    # OSMnx 2.x expects (west, south, east, north).
    bbox = (west, south, east, north)

    print("\nDELHI ROAD NETWORK DOWNLOAD")
    print("-" * 60)
    print(f"North: {north:.6f}")
    print(f"South: {south:.6f}")
    print(f"East:  {east:.6f}")
    print(f"West:  {west:.6f}")

    print("\nConnecting to OpenStreetMap...")
    print("Downloading drivable roads. Please keep the connection active.")

    graph = ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type="drive",
        simplify=True,
        retain_all=True
    )

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("Downloaded road network is empty.")

    # Save the actual road graph locally.
    ox.io.save_graphml(
        graph,
        filepath=ROAD_GRAPH_FILE
    )

    print("\nROAD NETWORK DOWNLOADED SUCCESSFULLY")
    print("-" * 60)
    print(f"Road nodes: {graph.number_of_nodes():,}")
    print(f"Directed road edges: {graph.number_of_edges():,}")
    print(f"Output file: {ROAD_GRAPH_FILE}")

    print("\nRoad network saved successfully.")


def main():
    print("=" * 65)
    print("DISASTER RELIEF VEHICLE ROUTING SYSTEM")
    print("Stage 2: OpenStreetMap Road Network Preparation")
    print("=" * 65)

    coordinates = load_coordinates()
    download_road_network(coordinates)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}")
        raise SystemExit(1)
