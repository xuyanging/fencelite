"""Exactness and complexity regressions for Method2 carrier routing."""

from __future__ import annotations

import math
from pathlib import Path
import random
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

from line_type_engine.method2 import text_family  # noqa: E402


def _geometry(
    index: int,
    start: tuple[float, float],
    end: tuple[float, float],
) -> text_family._CarrierGeometry:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    tangent = text_family.Point(dx / length, dy / length)
    return text_family._CarrierGeometry(
        index,
        (text_family.Point(*start), text_family.Point(*end)),
        (tangent, tangent),
        length,
    )


def _brute_graph(
    geometries: list[text_family._CarrierGeometry],
    tolerance: float,
) -> text_family._CarrierRouteGraph:
    adjacency: list[list[text_family._CarrierEdge]] = [
        [] for _ in geometries
    ]
    for left in range(len(geometries)):
        for right in range(left + 1, len(geometries)):
            gap, tangent = text_family._nearest_endpoint_connection(
                geometries[left], geometries[right]
            )
            if gap > tolerance:
                continue
            adjacency[left].append(text_family._CarrierEdge(right, gap, tangent))
            adjacency[right].append(text_family._CarrierEdge(left, gap, tangent))
    return text_family._CarrierRouteGraph(
        tuple(geometries),
        {geometry.op_index: index for index, geometry in enumerate(geometries)},
        tuple(tuple(edges) for edges in adjacency),
    )


def _early_exit_bridge(
    graph: text_family._CarrierRouteGraph,
    source_ops: set[int],
    target_ops: set[int],
    tolerance: float,
) -> tuple[text_family._CarrierGeometry, ...] | None:
    """The pre-optimization implementation, retained as a test oracle."""

    source_indices = [
        graph.index_by_op[index]
        for index in source_ops
        if index in graph.index_by_op
    ]
    targets = {
        graph.index_by_op[index]
        for index in target_ops
        if index in graph.index_by_op
    }
    if not source_indices or not targets:
        return None
    costs = [math.inf] * len(graph.geometries)
    previous = [-1] * len(graph.geometries)
    queue: list[tuple[float, int, int]] = []
    for index in source_indices:
        costs[index] = graph.geometries[index].length
        text_family.heapq.heappush(
            queue, (costs[index], graph.geometries[index].op_index, index)
        )
    reached = -1
    while queue:
        cost, _op_index, current = text_family.heapq.heappop(queue)
        if cost != costs[current]:
            continue
        if current in targets:
            reached = current
            break
        for edge in graph.adjacency[current]:
            turn_penalty = (1.0 - edge.tangent_similarity) * tolerance * 2.5
            next_cost = (
                cost
                + graph.geometries[edge.index].length
                + edge.gap * 8.0
                + turn_penalty
            )
            if next_cost >= costs[edge.index]:
                continue
            costs[edge.index] = next_cost
            previous[edge.index] = current
            text_family.heapq.heappush(
                queue,
                (next_cost, graph.geometries[edge.index].op_index, edge.index),
            )
    if reached < 0:
        return None
    route: list[text_family._CarrierGeometry] = []
    current = reached
    while current >= 0:
        route.append(graph.geometries[current])
        if current in source_indices:
            break
        current = previous[current]
    route.reverse()
    if not route or route[0].op_index not in source_ops:
        return None
    return tuple(route)


class CarrierRouteSpatialIndexTests(unittest.TestCase):
    def assert_matches_brute_force(
        self,
        geometries: list[text_family._CarrierGeometry],
        tolerance: float,
    ) -> None:
        self.assertEqual(
            text_family._carrier_route_graph_for(geometries, tolerance),
            _brute_graph(geometries, tolerance),
        )

    def test_randomized_graph_is_bit_exact(self) -> None:
        randomizer = random.Random(0xF3CE)
        for tolerance in (0.01, 0.2, 0.72, 2.5):
            geometries = []
            for index in range(160):
                start = (
                    randomizer.uniform(-40.0, 40.0),
                    randomizer.uniform(-40.0, 40.0),
                )
                angle = randomizer.uniform(-math.pi, math.pi)
                length = randomizer.uniform(0.01, 8.0)
                end = (
                    start[0] + math.cos(angle) * length,
                    start[1] + math.sin(angle) * length,
                )
                geometries.append(_geometry(index, start, end))
            self.assert_matches_brute_force(geometries, tolerance)

    def test_boundaries_negative_coordinates_and_duplicate_endpoints(self) -> None:
        geometries = [
            _geometry(0, (-2.0, -2.0), (-1.0, -1.0)),
            _geometry(1, (-1.0, -1.0), (0.0, 0.0)),
            _geometry(2, (1.0, 0.0), (2.0, 0.0)),
            _geometry(3, (2.0, 0.0), (3.0, 0.0)),
            _geometry(4, (3.0, 0.0), (4.0, 0.0)),
            _geometry(5, (-2.0, -2.0), (-3.0, -2.0)),
        ]
        self.assert_matches_brute_force(geometries, 1.0)
        self.assert_matches_brute_force(geometries, 0.0)

    def test_extreme_finite_coordinates_fall_back_without_overflow(self) -> None:
        point = text_family.Point
        tangent = (point(1.0, 0.0), point(1.0, 0.0))
        geometries = [
            text_family._CarrierGeometry(
                0, (point(1e300, 0.0), point(1e300, 1.0)), tangent, 1.0
            ),
            text_family._CarrierGeometry(
                1, (point(-1e300, 0.0), point(-1e300, 1.0)), tangent, 1.0
            ),
        ]
        self.assert_matches_brute_force(geometries, 1e-300)
        overflow_sum = [
            text_family._CarrierGeometry(
                0, (point(1.7e308, 0.0), point(1.7e308, 1.0)), tangent, 1.0
            ),
            text_family._CarrierGeometry(
                1, (point(-1.7e308, 0.0), point(-1.7e308, 1.0)), tangent, 1.0
            ),
        ]
        self.assert_matches_brute_force(overflow_sum, 1e308)

    def test_sparse_input_does_not_fall_back_to_quadratic_scan(self) -> None:
        geometries = [
            _geometry(index, (index * 10.0, 0.0), (index * 10.0 + 1.0, 0.0))
            for index in range(5_000)
        ]
        original = text_family._nearest_endpoint_connection
        with mock.patch.object(
            text_family,
            "_nearest_endpoint_connection",
            wraps=original,
        ) as nearest:
            graph = text_family._carrier_route_graph_for(geometries, 0.72)
        self.assertEqual(nearest.call_count, 0)
        self.assertTrue(all(not edges for edges in graph.adjacency))

    def test_reused_shortest_paths_match_every_early_exit_query(self) -> None:
        randomizer = random.Random(0xD1A57A)
        geometries = []
        for index in range(180):
            start = (
                randomizer.uniform(-12.0, 12.0),
                randomizer.uniform(-12.0, 12.0),
            )
            angle = randomizer.uniform(-math.pi, math.pi)
            length = randomizer.uniform(0.1, 3.0)
            geometries.append(_geometry(
                10_000 + index,
                start,
                (
                    start[0] + math.cos(angle) * length,
                    start[1] + math.sin(angle) * length,
                ),
            ))
        tolerance = 1.4
        graph = text_family._carrier_route_graph_for(geometries, tolerance)
        operation_ids = [geometry.op_index for geometry in geometries]
        for _case in range(40):
            source_ops = set(randomizer.sample(operation_ids, 3))
            paths = text_family._carrier_shortest_paths(
                graph, source_ops, tolerance
            )
            for _query in range(12):
                target_ops = set(randomizer.sample(operation_ids, 4))
                expected = _early_exit_bridge(
                    graph, source_ops, target_ops, tolerance
                )
                actual = text_family._carrier_bridge_from_paths(
                    graph, paths, source_ops, target_ops
                )
                self.assertEqual(actual, expected)

    def test_reused_paths_preserve_equal_cost_heap_ties(self) -> None:
        geometries = [
            _geometry(10, (0.0, 0.0), (1.0, 0.0)),
            _geometry(20, (1.0, 0.0), (2.0, 0.0)),
            _geometry(30, (1.0, 0.0), (1.0, 1.0)),
            _geometry(40, (2.0, 0.0), (3.0, 0.0)),
            _geometry(50, (1.0, 1.0), (1.0, 2.0)),
        ]
        tolerance = 0.0
        graph = text_family._carrier_route_graph_for(geometries, tolerance)
        source_ops = {10}
        paths = text_family._carrier_shortest_paths(graph, source_ops, tolerance)
        for targets in ({40, 50}, {50, 40}, {20, 30}, {10, 50}):
            self.assertEqual(
                text_family._carrier_bridge_from_paths(
                    graph, paths, source_ops, set(targets)
                ),
                _early_exit_bridge(graph, source_ops, set(targets), tolerance),
            )

    def test_negative_step_uses_historical_early_exit_semantics(self) -> None:
        geometries = (
            _geometry(10, (0.0, 0.0), (1.0, 0.0)),
            _geometry(20, (2.0, 0.0), (3.0, 0.0)),
            _geometry(30, (4.0, 0.0), (6.0, 0.0)),
        )
        edge = text_family._CarrierEdge
        graph = text_family._CarrierRouteGraph(
            geometries,
            {10: 0, 20: 1, 30: 2},
            (
                (edge(1, 0.0, 1.0), edge(2, 0.0, 1.0)),
                (edge(0, 0.0, 1.0), edge(2, 0.0, 3.0)),
                (edge(0, 0.0, 1.0), edge(1, 0.0, 3.0)),
            ),
        )
        tolerance = 1.0
        self.assertFalse(text_family._carrier_steps_can_be_reused(
            graph, tolerance
        ))
        expected = _early_exit_bridge(graph, {10}, {20}, tolerance)
        self.assertEqual([item.op_index for item in expected or ()], [10, 20])
        self.assertEqual(
            text_family._shortest_carrier_bridge(
                graph, {10}, {20}, tolerance
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
