"""

Users enter real location names (or IDs such as P3 / DEPOT) directly.
The route is calculated with Dijkstra using OpenStreetMap road lengths.
"""
from __future__ import annotations

import csv
import re
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from typing import Any

import networkx as nx

from algorithms import build_adjacency, dijkstra, reconstruct_path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOCATIONS_FILE = DATA_DIR / "locations.csv"
SNAPPED_FILE = DATA_DIR / "location_road_nodes.csv"
ROAD_GRAPH_FILE = DATA_DIR / "delhi_north_drive_network.graphml"
ALTERNATE_ROAD_GRAPH_FILE = DATA_DIR / "delhi_north_drive.graphml"
NUMBER_OF_ALTERNATIVE_ROUTES = 3
YEN_CANDIDATE_POOL_SIZE = 40
MAX_ALTERNATIVE_DETOUR_FACTOR = 3.0
DIVERSITY_OVERLAP_THRESHOLDS = (0.70, 0.80, 0.90)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file and return its rows as dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _normalise_column_name(value: str) -> str:
    """Normalise CSV headers so common naming variations are accepted."""
    value = (value or "").replace("\ufeff", "").strip().casefold()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def validate_locations(
    locations: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Validate locations and standardise common CSV header variations.

    Accepted ID headers include id, location_id, loc_id, code and location_code.
    Accepted name headers include name, location_name and site_name.
    Accepted role headers include role, type, location_type and category.
    """
    if not locations:
        raise ValueError("locations.csv is empty.")

    aliases = {
        "id": ("id", "location_id", "loc_id", "code", "location_code", "site_id", "point_id"),
        "name": ("name", "location_name", "site_name", "facility_name", "location"),
        "role": ("role", "location_role", "type", "location_type", "category", "site_type"),
    }

    original_headers = list(locations[0].keys())
    normalised_to_original = {
        _normalise_column_name(header): header for header in original_headers
    }
    selected_headers: dict[str, str] = {}
    for standard_name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalised_to_original:
                selected_headers[standard_name] = normalised_to_original[candidate]
                break

    missing_columns = set(aliases) - set(selected_headers)
    if missing_columns:
        found = ", ".join(repr(header) for header in original_headers)
        raise ValueError(
            "locations.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
            + f". Found headers: {found}.\n"
            + "Accepted ID headers: id, location_id, loc_id, code, location_code. "
            + "Accepted name headers: name, location_name, site_name. "
            + "Accepted role headers: role, type, location_type, category."
        )

    # Create canonical rows so the rest of the application always uses id/name/role.
    canonical_rows: list[dict[str, str]] = []
    for raw_row in locations:
        row = dict(raw_row)
        for standard_name, original_header in selected_headers.items():
            row[standard_name] = (raw_row.get(original_header) or "").strip()

        role = row["role"].strip().upper().replace("-", "_").replace(" ", "_")
        if role in {"DELIVERY", "DESTINATION", "RELIEF_SITE", "SCHOOL", "DELIVERYLOCATION"}:
            role = "DELIVERY_POINT"
        row["role"] = role
        canonical_rows.append(row)

    by_id: dict[str, dict[str, str]] = {}
    for row in canonical_rows:
        location_id = row["id"]
        name = row["name"]
        if not location_id or not name:
            raise ValueError(f"A location row has a blank ID or name: {row}")
        if location_id in by_id:
            raise ValueError(f"Duplicate location ID in locations.csv: {location_id}")
        by_id[location_id] = row

    depots = {
        location_id: row
        for location_id, row in by_id.items()
        if row["role"] == "DEPOT"
    }
    destinations = {
        location_id: row
        for location_id, row in by_id.items()
        if row["role"] == "DELIVERY_POINT"
    }

    if len(depots) != 1:
        role_values = sorted({row["role"] for row in by_id.values()})
        raise ValueError(
            "locations.csv must contain exactly one row with role DEPOT. "
            f"Roles found: {role_values}"
        )
    if not destinations:
        role_values = sorted({row["role"] for row in by_id.values()})
        raise ValueError(
            "locations.csv must contain at least one DELIVERY_POINT row. "
            f"Roles found: {role_values}. Use DELIVERY_POINT for delivery locations."
        )

    return depots, destinations


def load_snapped_nodes() -> dict[str, int]:
    """Load the mapping from project location IDs to OSM road-node IDs."""
    rows = read_csv(SNAPPED_FILE)
    result: dict[str, int] = {}

    required = {"location_id", "road_node_id"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(
            "location_road_nodes.csv must contain location_id and road_node_id columns."
        )

    for row in rows:
        location_id = (row.get("location_id") or "").strip()
        road_node_text = (row.get("road_node_id") or "").strip()
        if not location_id or not road_node_text:
            raise ValueError(f"Invalid row in location_road_nodes.csv: {row}")
        if location_id in result:
            raise ValueError(f"Duplicate location mapping for ID: {location_id}")
        try:
            result[location_id] = int(road_node_text)
        except ValueError as exc:
            raise ValueError(
                f"Road node ID for {location_id} is not a valid integer: {road_node_text}"
            ) from exc

    return result


def load_road_graph() -> nx.MultiDiGraph | nx.DiGraph:
    """Load the saved directed OpenStreetMap road graph."""
    graph_path = ROAD_GRAPH_FILE
    if not graph_path.exists() and ALTERNATE_ROAD_GRAPH_FILE.exists():
        graph_path = ALTERNATE_ROAD_GRAPH_FILE

    if not graph_path.exists():
        raise FileNotFoundError(
            "Road graph is missing. Expected one of these files:\n"
            f"  {ROAD_GRAPH_FILE}\n  {ALTERNATE_ROAD_GRAPH_FILE}\n"
            "Run this first from the project folder:\n"
            "python scripts/prepare_road_network.py"
        )

    # OSM road-node IDs are numeric OpenStreetMap identifiers.
    graph = nx.read_graphml(graph_path, node_type=int)
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("The saved road graph is empty. Prepare the road network again.")

    return graph


def normalise_text(value: str) -> str:
    """Make matching independent of case and repeated spaces/punctuation."""
    value = value.casefold().strip()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def resolve_location(
    user_input: str,
    by_id: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str]] | None:
    """Match a typed location ID/name, allowing a unique partial name."""
    query = normalise_text(user_input)
    if not query:
        return None

    # First prefer an exact ID or exact location-name match.
    exact_matches: list[tuple[str, dict[str, str]]] = []
    for location_id, row in by_id.items():
        if query == normalise_text(location_id) or query == normalise_text(row["name"]):
            exact_matches.append((location_id, row))

    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        return None

    # Also accept a partial name only when it identifies one location uniquely.
    partial_matches = [
        (location_id, row)
        for location_id, row in by_id.items()
        if query in normalise_text(row["name"])
        or query in normalise_text(location_id)
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def prompt_for_location(
    prompt: str,
    by_id: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str]]:
    """Keep prompting until a valid name or location ID is entered."""
    while True:
        user_input = input(prompt).strip()
        selected = resolve_location(user_input, by_id)
        if selected is not None:
            return selected

        print("Location not recognised or the name is ambiguous.")
        print("Please enter the full location name or its ID from the list above.")


def show_available_locations(by_id: dict[str, dict[str, str]]) -> None:
    """Print locations as names and IDs; no numbered location choices."""
    print("\nAVAILABLE LOCATIONS")
    print("-" * 78)
    for location_id, row in by_id.items():
        role = (row.get("role") or "").strip()
        print(f"{row['name']} [{location_id}]  -  {role}")
    print("-" * 78)
    print("You can enter a location name (recommended) or its ID, such as P3 or DEPOT.")


def display_distance_table(
    depot_id: str,
    depot: dict[str, str],
    destinations: dict[str, dict[str, str]],
    snapped: dict[str, int],
    adjacency: dict[Any, list[tuple[Any, float]]],
) -> None:
    """Display shortest directed road distances from the depot."""
    source = snapped[depot_id]
    distances, predecessors = dijkstra(adjacency, source)

    print("\nSHORTEST ROAD DISTANCES FROM THE DEPOT")
    print("=" * 104)
    print(f"Depot: {depot['name']} [{depot_id}]")
    print("Distance uses OpenStreetMap road-edge lengths, not live traffic or travel time.")
    print("-" * 104)
    print(f"{'ID':<8} {'DELIVERY LOCATION':<65} {'DISTANCE (KM)':>15} {'PATH NODES':>12}")
    print("-" * 104)

    unreachable: list[str] = []
    for location_id, place in destinations.items():
        target = snapped[location_id]
        distance_m = distances.get(target)
        if distance_m is None:
            unreachable.append(location_id)
            print(
                f"{location_id:<8} {place['name'][:64]:<65} "
                f"{'UNREACHABLE':>15} {'-':>12}"
            )
            continue

        path = reconstruct_path(predecessors, source, target)
        print(
            f"{location_id:<8} {place['name'][:64]:<65} "
            f"{distance_m / 1000:>15.3f} {len(path):>12,}"
        )

    print("-" * 104)
    if unreachable:
        print("Unreachable location IDs:", ", ".join(unreachable))
    print("These are shortest mapped road lengths, not traffic-aware estimates.")


def simplify_adjacency(
    adjacency: dict[Any, list[tuple[Any, float]]],
) -> dict[Any, list[tuple[Any, float]]]:
    """Keep only the shortest edge between each ordered pair of road nodes.

    OSMnx graphs can contain parallel edges between the same two nodes.
    For node-route comparison, retaining the minimum-length parallel edge
    prevents duplicate alternatives that have exactly the same node sequence.
    """
    simple: dict[Any, list[tuple[Any, float]]] = {}
    for source, neighbours in adjacency.items():
        best_edge: dict[Any, float] = {}
        for destination, weight in neighbours:
            weight = float(weight)
            if destination not in best_edge or weight < best_edge[destination]:
                best_edge[destination] = weight
        simple[source] = list(best_edge.items())
    return simple


def constrained_dijkstra_path(
    adjacency: dict[Any, list[tuple[Any, float]]],
    source: Any,
    target: Any,
    banned_nodes: set[Any] | None = None,
    banned_edges: set[tuple[Any, Any]] | None = None,
) -> list[Any] | None:
    """Find one shortest path while temporarily excluding nodes/edges.

    This is a Dijkstra search used by Yen's K-shortest loopless-path method.
    Exclusions let the algorithm generate different routes instead of
    repeatedly returning the same shortest path.
    """
    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()

    if source not in adjacency or target not in adjacency:
        return None
    if source in banned_nodes or target in banned_nodes:
        return None

    distances: dict[Any, float] = {source: 0.0}
    previous: dict[Any, Any] = {}
    queue: list[tuple[float, Any]] = [(0.0, source)]

    while queue:
        current_distance, current = heappop(queue)
        if current_distance != distances.get(current):
            continue
        if current == target:
            break

        for neighbour, weight in adjacency.get(current, []):
            if neighbour in banned_nodes:
                continue
            if (current, neighbour) in banned_edges:
                continue

            candidate_distance = current_distance + weight
            if candidate_distance < distances.get(neighbour, float("inf")):
                distances[neighbour] = candidate_distance
                previous[neighbour] = current
                heappush(queue, (candidate_distance, neighbour))

    if target not in distances:
        return None

    path = [target]
    current = target
    while current != source:
        current = previous.get(current)
        if current is None:
            return None
        path.append(current)
    path.reverse()
    return path


def calculate_path_distance(
    adjacency: dict[Any, list[tuple[Any, float]]], path: list[Any]
) -> float:
    """Calculate a path's total mapped road length in metres."""
    edge_weights = {
        source: {destination: weight for destination, weight in neighbours}
        for source, neighbours in adjacency.items()
    }
    return sum(edge_weights[u][v] for u, v in zip(path, path[1:]))


def yen_k_shortest_paths(
    adjacency: dict[Any, list[tuple[Any, float]]],
    source: Any,
    target: Any,
    limit: int = NUMBER_OF_ALTERNATIVE_ROUTES,
) -> list[tuple[list[Any], float]]:
    """Return up to `limit` distinct shortest loopless routes, in order.

    Route 1 is calculated with the project's Dijkstra implementation.
    Further distinct alternatives are generated with Yen's algorithm, which
    repeatedly runs constrained Dijkstra searches for spur paths.
    """
    if limit < 1:
        return []
    if source not in adjacency or target not in adjacency:
        return []
    if source == target:
        return [([source], 0.0)]

    # The first route is the standard Dijkstra shortest path used by the project.
    distances, predecessors = dijkstra(adjacency, source, target)
    shortest_distance = distances.get(target)
    first_path = reconstruct_path(predecessors, source, target)
    if shortest_distance is None or not first_path:
        return []

    accepted_paths: list[list[Any]] = [first_path]
    accepted_path_keys = {tuple(first_path)}
    accepted_routes: list[tuple[list[Any], float]] = [(first_path, float(shortest_distance))]

    candidate_heap: list[tuple[float, int, list[Any]]] = []
    candidate_keys: set[tuple[Any, ...]] = set()
    sequence = count()

    for _ in range(1, limit):
        previous_path = accepted_paths[-1]

        # Try every node in the previous route as a spur point.
        for spur_index in range(len(previous_path) - 1):
            root_path = previous_path[: spur_index + 1]
            spur_node = root_path[-1]
            banned_edges: set[tuple[Any, Any]] = set()

            # Remove the next edge of every accepted route sharing this root.
            for route in accepted_paths:
                if (
                    len(route) > spur_index + 1
                    and route[: spur_index + 1] == root_path
                ):
                    banned_edges.add((route[spur_index], route[spur_index + 1]))

            # A loopless route cannot revisit an earlier node in its root path.
            banned_nodes = set(root_path[:-1])
            spur_path = constrained_dijkstra_path(
                adjacency,
                spur_node,
                target,
                banned_nodes=banned_nodes,
                banned_edges=banned_edges,
            )
            if not spur_path:
                continue

            total_path = root_path[:-1] + spur_path
            path_key = tuple(total_path)
            if path_key in accepted_path_keys or path_key in candidate_keys:
                continue

            total_distance = calculate_path_distance(adjacency, total_path)
            heappush(candidate_heap, (total_distance, next(sequence), total_path))
            candidate_keys.add(path_key)

        if not candidate_heap:
            break

        next_distance, _, next_path = heappop(candidate_heap)
        accepted_paths.append(next_path)
        accepted_path_keys.add(tuple(next_path))
        accepted_routes.append((next_path, next_distance))

    return accepted_routes


def shared_road_overlap_ratio(
    adjacency: dict[Any, list[tuple[Any, float]]],
    path_a: list[Any],
    distance_a: float,
    path_b: list[Any],
    distance_b: float,
) -> float:
    """Return the fraction of the shorter route's distance shared by both.

    Edges are directed, so A -> B and B -> A are treated as different roads.
    The result is between 0.0 (no shared road edges) and 1.0 (all of the
    shorter route is shared).
    """
    if min(distance_a, distance_b) <= 0:
        return 0.0

    weights = {
        (source, destination): float(weight)
        for source, neighbours in adjacency.items()
        for destination, weight in neighbours
    }
    edges_a = set(zip(path_a, path_a[1:]))
    edges_b = set(zip(path_b, path_b[1:]))
    common_edges = edges_a & edges_b
    common_distance = sum(weights.get(edge, 0.0) for edge in common_edges)
    return min(1.0, common_distance / min(distance_a, distance_b))


def generate_route_candidates(
    adjacency: dict[Any, list[tuple[Any, float]]],
    source: Any,
    target: Any,
    shortest_path: list[Any],
) -> list[tuple[list[Any], float]]:
    """Generate alternatives using Yen plus deliberate bypasses of route sections.

    Yen supplies the next shortest loop-free paths. The additional searches
    temporarily remove sections of the shortest route, encouraging Dijkstra
    to find paths that use other streets instead of tiny variations of the
    same corridor. Reported distances always use original OSM edge lengths.
    """
    candidates: dict[tuple[Any, ...], tuple[list[Any], float]] = {}

    def add_candidate(path: list[Any] | None) -> None:
        if not path or path[0] != source or path[-1] != target:
            return
        key = tuple(path)
        if key == tuple(shortest_path) or key in candidates:
            return
        try:
            distance = calculate_path_distance(adjacency, path)
        except KeyError:
            return
        candidates[key] = (path, distance)

    # Keep a reasonably broad pool of mathematically short alternatives.
    for path, _distance in yen_k_shortest_paths(
        adjacency, source, target, limit=YEN_CANDIDATE_POOL_SIZE
    ):
        add_candidate(path)

    # Also force detours around different sections of the shortest corridor.
    shortest_edges = list(zip(shortest_path, shortest_path[1:]))
    edge_count = len(shortest_edges)
    if edge_count:
        fractions = (0.20, 0.35, 0.50, 0.70, 1.00)
        for fraction in fractions:
            block_size = max(1, min(edge_count, round(edge_count * fraction)))
            last_start = edge_count - block_size
            starts = {
                0,
                last_start,
                round(last_start * 0.25),
                round(last_start * 0.50),
                round(last_start * 0.75),
            }
            for start_index in sorted(starts):
                blocked_section = set(
                    shortest_edges[start_index : start_index + block_size]
                )
                alternative = constrained_dijkstra_path(
                    adjacency,
                    source,
                    target,
                    banned_edges=blocked_section,
                )
                add_candidate(alternative)

    return sorted(candidates.values(), key=lambda item: item[1])


def select_diverse_routes(
    adjacency: dict[Any, list[tuple[Any, float]]],
    shortest_path: list[Any],
    shortest_distance: float,
    candidates: list[tuple[list[Any], float]],
    limit: int = NUMBER_OF_ALTERNATIVE_ROUTES,
) -> list[tuple[list[Any], float]]:
    """Select alternatives with meaningful road differences from each other.

    Near-identical alternatives are filtered out. Similarity is relaxed in
    small steps (70%, 80%, then 90%) only when the network has fewer diverse
    choices. Alternatives beyond three times the shortest distance are not
    shown as practical comparison routes.
    """
    selected: list[tuple[list[Any], float]] = [(shortest_path, shortest_distance)]
    if limit <= 1 or shortest_distance <= 0:
        return selected[:limit]

    max_distance = shortest_distance * MAX_ALTERNATIVE_DETOUR_FACTOR
    remaining = [
        (path, distance)
        for path, distance in candidates
        if shortest_distance < distance <= max_distance
    ]

    for allowed_overlap in DIVERSITY_OVERLAP_THRESHOLDS:
        while len(selected) < limit:
            eligible: list[tuple[float, float, list[Any], float]] = []
            for path, distance in remaining:
                overlaps = [
                    shared_road_overlap_ratio(
                        adjacency, path, distance, chosen_path, chosen_distance
                    )
                    for chosen_path, chosen_distance in selected
                ]
                maximum_overlap = max(overlaps, default=0.0)
                if maximum_overlap <= allowed_overlap:
                    eligible.append((maximum_overlap, distance, path, distance))

            if not eligible:
                break

            # Prefer the most road-diverse option; break ties by shorter length.
            _overlap, _distance, chosen_path, chosen_distance = min(
                eligible, key=lambda item: (item[0], item[1])
            )
            selected.append((chosen_path, chosen_distance))
            chosen_key = tuple(chosen_path)
            remaining = [item for item in remaining if tuple(item[0]) != chosen_key]

        if len(selected) >= limit:
            break

    return selected


def display_pair_route(
    by_id: dict[str, dict[str, str]],
    snapped: dict[str, int],
    adjacency: dict[Any, list[tuple[Any, float]]],
) -> None:
    """Compare road-diverse alternatives and select the minimum-distance route."""
    show_available_locations(by_id)

    start_id, start_place = prompt_for_location(
        "\nEnter START location name or ID: ", by_id
    )
    destination_id, destination_place = prompt_for_location(
        "Enter DESTINATION location name or ID: ", by_id
    )

    print("\nSelected route:")
    print(f"Start       : {start_place['name']} [{start_id}]")
    print(f"Destination : {destination_place['name']} [{destination_id}]")

    if start_id == destination_id:
        print("\nStart and destination are the same location. Choose two different locations.")
        return

    source_node = snapped[start_id]
    target_node = snapped[destination_id]

    if source_node not in adjacency:
        print(f"\nThe mapped road node for {start_id} is not present in the road graph.")
        return
    if target_node not in adjacency:
        print(f"\nThe mapped road node for {destination_id} is not present in the road graph.")
        return

    print("\nFinding the shortest route and road-diverse alternatives...")
    route_adjacency = simplify_adjacency(adjacency)

    # Dijkstra remains the authority for the globally shortest route.
    distances, predecessors = dijkstra(route_adjacency, source_node, target_node)
    shortest_distance = distances.get(target_node)
    shortest_path = reconstruct_path(predecessors, source_node, target_node)

    if shortest_distance is None or not shortest_path:
        print("\nNo directed road path was found between these locations.")
        print("The road graph respects one-way road directions.")
        return

    candidates = generate_route_candidates(
        route_adjacency, source_node, target_node, shortest_path
    )
    routes = select_diverse_routes(
        route_adjacency,
        shortest_path,
        float(shortest_distance),
        candidates,
        limit=NUMBER_OF_ALTERNATIVE_ROUTES,
    )

    print("\nROUTE COMPARISON")
    print("=" * 104)
    print("Route 1 is the shortest route calculated by Dijkstra.")
    print(
        "Alternatives are selected to use different directed road links; "
        "near-identical routes are filtered out."
    )
    print(
        f"Alternative search limit: up to {MAX_ALTERNATIVE_DETOUR_FACTOR:.1f}x "
        "the shortest distance."
    )
    print("-" * 104)
    print(
        f"{'ROUTE':<10} {'DISTANCE':>13} {'LONGER BY':>13} "
        f"{'SHARED WITH ROUTE 1':>22}  DECISION"
    )
    print("-" * 104)

    for index, (path, distance_m) in enumerate(routes, start=1):
        extra_m = max(0.0, distance_m - shortest_distance)
        extra_pct = (extra_m / shortest_distance * 100) if shortest_distance > 0 else 0.0
        if index == 1:
            longer_text = "-"
            shared_text = "100.0%"
            decision = "SELECTED SHORTEST"
        else:
            overlap = shared_road_overlap_ratio(
                route_adjacency,
                path,
                distance_m,
                shortest_path,
                float(shortest_distance),
            )
            longer_text = f"+{extra_m / 1000:.3f} km ({extra_pct:.1f}%)"
            shared_text = f"{overlap * 100:.1f}%"
            decision = "DIVERSE ALTERNATIVE"

        print(
            f"Route {index:<4} {distance_m / 1000:>10.3f} km "
            f"{longer_text:>13} {shared_text:>22}  {decision}"
        )

    print("-" * 104)
    if len(routes) == 1:
        print(
            "No sufficiently different alternative route was found in the saved "
            "road network within the configured distance limit."
        )
        print(
            "The previously seen alternatives were almost the same road path, "
            "so they are intentionally not displayed as useful alternatives."
        )
    else:
        longest_index, longest_route = max(
            enumerate(routes, start=1), key=lambda item: item[1][1]
        )
        print(
            f"Longest displayed alternative: Route {longest_index} "
            f"({longest_route[1] / 1000:.3f} km)."
        )
        print(
            f"Shortest route selected: Route 1 "
            f"({shortest_distance / 1000:.3f} km)."
        )

    print("\nSELECTED SHORTEST ROUTE DETAILS")
    print("-" * 104)
    print(
        f"Road distance : {shortest_distance / 1000:.3f} km "
        f"({shortest_distance:,.1f} metres)"
    )
    print(f"Road nodes    : {len(shortest_path):,}")
    print("OpenStreetMap road-node ID sequence:")
    if len(shortest_path) <= 60:
        print(" -> ".join(map(str, shortest_path)))
    else:
        print(" -> ".join(map(str, shortest_path[:25])))
        print(f"... {len(shortest_path) - 50:,} intermediate node IDs omitted ...")
        print(" -> ".join(map(str, shortest_path[-25:])))

    print("\nWHY THIS ROUTE IS SELECTED")
    print(
        "Dijkstra selects the route with the minimum total mapped road distance. "
        "The other displayed routes use different road links and are shown only "
        "for comparison."
    )
    print(
        "Important: this uses a saved OpenStreetMap road snapshot and road lengths "
        "only. It does not detect live traffic, road damage, closures, or emergency "
        "vehicle restrictions."
    )


def main() -> None:
    print("=" * 86)
    print("DISASTER RELIEF VEHICLE ROUTING SYSTEM")
    print("Shortest + road-diverse alternatives on Delhi OpenStreetMap roads")
    print("=" * 86)

    locations = read_csv(LOCATIONS_FILE)
    depots, destinations = validate_locations(locations)
    depot_id, depot = next(iter(depots.items()))
    # validate_locations returns canonical rows with standard id/name/role keys.
    by_id = {**depots, **destinations}

    snapped = load_snapped_nodes()
    missing_ids = set(by_id) - set(snapped)
    if missing_ids:
        raise ValueError(
            "These locations are not mapped to road nodes: "
            + ", ".join(sorted(missing_ids))
            + ". Re-run: python scripts/prepare_road_network.py"
        )

    print("\nLoading saved Delhi road graph...")
    graph = load_road_graph()
    print(
        f"Road graph loaded: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} directed edges"
    )
    print(f"Mapped study locations: {len(snapped)}")
    print("Building weighted adjacency list...")
    adjacency = build_adjacency(graph)

    # Check all mapped OSM nodes before accepting route requests.
    absent_nodes = [
        f"{location_id} -> {snapped[location_id]}"
        for location_id in by_id
        if snapped[location_id] not in adjacency
    ]
    if absent_nodes:
        raise ValueError(
            "Some mapped road nodes are absent from the saved graph: "
            + "; ".join(absent_nodes)
            + ". Re-run the road-network preparation script."
        )

    print("Dijkstra is ready. Edge weight = OSM road length in metres.")

    while True:
        print("\nMAIN MENU")
        print("1. Compare road-diverse routes and select the shortest")
        print("2. Show shortest distances from the depot")
        print("3. Exit")
        choice = input("Choose an option (1-3): ").strip()

        if choice == "1":
            display_pair_route(by_id, snapped, adjacency)
        elif choice == "2":
            display_distance_table(depot_id, depot, destinations, snapped, adjacency)
        elif choice == "3":
            print("Exiting Disaster Relief Vehicle Routing System.")
            break
        else:
            print("Invalid choice. Please enter 1, 2 or 3.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
