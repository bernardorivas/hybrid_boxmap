"""Tests for plotting native CMGDB Atlas Morse boxes."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hybrid_dynamics import (
    ATLAS_MORSE_PLOT_SCHEMA,
    PlotHybridMorseSets,
    SuspensionAtlasCharts,
    atlas_morse_hasse,
    extract_atlas_morse_plot_data,
    load_atlas_finite_relation_index_annotations,
    load_atlas_morse_plot_data,
    save_atlas_morse_plot_data,
    save_hybrid_morse_figure,
)
from hybrid_dynamics.src.hybrid_morse_plot import morse_graph_node_labels


class _FakeAtlasMorseGraph:
    def __init__(self) -> None:
        self._boxes = {
            0: (
                (0, (-0.2, 0.1, 0.0, 0.3)),
                (0, (0.0, 0.1, 0.2, 0.3)),
                (1, (0.1, 0.0, 0.3, 0.5)),
            ),
            1: ((0, (-0.05, -0.05, 0.05, 0.05)),),
            2: ((0, (0.35, 0.55, 0.45, 0.65)),),
        }

    def vertices(self):
        return (0, 1, 2)

    def edges(self):
        return ((2, 1), (1, 0))

    def morse_set_chart_boxes(self, node):
        return self._boxes[node]


def _charts() -> SuspensionAtlasCharts:
    return SuspensionAtlasCharts(
        base_bounds=((-1.0, 1.0), (-1.0, 1.0)),
        guard_bounds=((-1.0, 1.0),),
        guard_coordinates=lambda point: (point[1],),
        guard_embedding=lambda intrinsic: (1.0, intrinsic[0]),
    )


def _finite_relation_candidate(node: int, shift_class: list[str]) -> dict[str, object]:
    cell_counts = [1, 0, 0]
    return {
        "candidate": f"test-M{node}",
        "model": "test-atlas",
        "depth": 4,
        "t_star": 2.0,
        "finite_relation_blockers": [],
        "continuous_system_conley_index_certified": False,
        "whole_cell_outer_enclosure_certified": False,
        "analytic_conley_label_attached": False,
        "finite_relation_shift_class": shift_class,
        "top_cell_pair": {
            "morse_node": node,
            "pair_invariance_passed": True,
            "combinatorial_isolation_passed": True,
            "s_is_strongly_connected": True,
            "first_condition_violations": [],
            "second_condition_violations": [],
            "recurrent_components_not_s_or_a": [],
            "empty_s_sources": [],
            "unevaluated_S_sources": [],
            "unevaluated_X_sources": [],
        },
        "relation_provenance": {
            "X_sources_evaluated": 1,
            "X_sources_total": 1,
            "unresolved_event_stage_edges": 0,
            "original_exit_edges_retained": True,
        },
        "quotient_nerve": {
            "finite_intersections_verified_contractible": True,
            "boundary_squared_zero_validated": True,
            "vertices_are_actual_atlas_boxes": True,
            "analytic_orbit_skeleton_used": False,
            "cells_by_dimension": cell_counts,
        },
        "cellular_pair": {
            "P0_subset_P1": True,
            "carrier_preserves_pair": True,
            "selected_chain_map_preserves_P1": True,
            "selected_chain_map_preserves_P0": True,
            "P1_cells_by_dimension": cell_counts,
            "relative_basis_by_dimension": cell_counts,
        },
        "carrier_certificate": {
            "all_carrier_values_acyclic": True,
            "face_nesting_validated": True,
            "selected_chain_map_subordinate": True,
            "chain_equation_dF_equals_Fd": True,
            "coefficient_field": "GF(5)",
            "chain_equation_failure_witnesses": [],
        },
        "finite_relation_conley_index": {
            "validation": {
                "matrix_shapes_and_entries": True,
                "boundary_squared_zero": True,
                "chain_map_equation": True,
            },
            "coefficient_field": 5,
            "result_scope": "finite_reset_quotient_relation",
            "finite_relation_algebra_validated": True,
            "continuous_system_conley_index_certified": False,
            "cell_counts": cell_counts,
            "shift_class": shift_class,
        },
    }


def _finite_relation_audit() -> dict[str, object]:
    shifts = {
        0: ["x-1", "x-1", "0"],
        1: ["0", "x-1", "0"],
        2: ["0", "0", "x^2+1"],
    }
    return {
        "schema": "physical-conley-finite-relation-audit-v2",
        "certified_continuous_system_indices": 0,
        "computed_finite_relation_indices": 3,
        "candidates": [
            _finite_relation_candidate(node, shift)
            for node, shift in shifts.items()
        ],
    }


def test_extract_round_trip_preserves_chart_tags_boxes_and_morse_order(tmp_path):
    data = extract_atlas_morse_plot_data(
        _FakeAtlasMorseGraph(),
        _charts(),
        metadata={"relation_scope": "nonempty local-window relation", "t_star": 2.0},
    )

    assert data.vertex_ids == (0, 1, 2)
    assert data.edges == ((1, 0), (2, 1))
    assert len(data.nodes[0].boxes) == 3
    assert {box.chart_id for box in data.nodes[0].boxes} == {0, 1}

    output = save_atlas_morse_plot_data(data, tmp_path / "atlas.json")
    assert ATLAS_MORSE_PLOT_SCHEMA in output.read_text(encoding="utf-8")
    assert load_atlas_morse_plot_data(output) == data


def test_plot_hybrid_morse_sets_dispatches_to_native_atlas_path():
    plot = PlotHybridMorseSets(
        _FakeAtlasMorseGraph(),
        atlas_charts=_charts(),
        axis_labels=(r"$q$", r"$v$"),
        show_component_sizes=True,
    )
    try:
        assert len(plot.projection_axes) == 1
        assert plot.handle_axes == ()
        assert plot.handle_axis is None
        assert plot.morse_graph_axis is not None
        assert set(plot.morse_graph.nodes) == {0, 1, 2}
        assert set(plot.morse_graph.edges) == {(2, 1), (1, 0)}
        assert plot.projection_axes[0].get_xlabel() == r"$q$"
        assert len(plot.projection_axes[0].collections) == 3
        assert set(morse_graph_node_labels(plot.morse_graph_axis)) == {
            "M(0)\n3 cells",
            "M(1)\n1 cells",
            "M(2)\n1 cells",
        }
    finally:
        plt.close(plot.figure)


def test_certificate_backed_finite_relation_labels_are_scoped_and_plotted(tmp_path):
    data = extract_atlas_morse_plot_data(
        _FakeAtlasMorseGraph(),
        _charts(),
        metadata={"model": "test-atlas", "depth": 4, "t_star": 2.0},
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(_finite_relation_audit()), encoding="utf-8")

    annotations = load_atlas_finite_relation_index_annotations(audit_path, data)
    plot = PlotHybridMorseSets(
        data,
        finite_relation_annotations=annotations,
        show_status_note=True,
    )
    try:
        assert annotations.shift_classes[0] == ("x-1", "x-1", "0")
        assert annotations.continuous_system_conley_index_certified is False
        graph_labels = set(morse_graph_node_labels(plot.morse_graph_axis))
        assert "M(0)\n(x-1, x-1, 0)" in graph_labels
        assert "M(1)\n(0, x-1, 0)" in graph_labels
        assert any(
            "finite sampled-relation Conley index" in text.get_text()
            and "continuous-system certification not established" in text.get_text()
            for text in plot.figure.texts
        )
    finally:
        plt.close(plot.figure)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda audit: audit["candidates"][0]["carrier_certificate"].update(
                {"all_carrier_values_acyclic": False}
            ),
            "carrier.*certificate flags",
        ),
        (
            lambda audit: audit["candidates"][0].update(
                {"finite_relation_blockers": ["not ready"]}
            ),
            "unresolved finite-relation blockers",
        ),
        (
            lambda audit: audit["candidates"][0].update(
                {"continuous_system_conley_index_certified": True}
            ),
            "must not claim continuous-system certification",
        ),
    ),
)
def test_finite_relation_label_loader_rejects_tampered_audits(
    tmp_path,
    mutate,
    message,
):
    data = extract_atlas_morse_plot_data(
        _FakeAtlasMorseGraph(),
        _charts(),
        metadata={"model": "test-atlas", "depth": 4, "t_star": 2.0},
    )
    audit = copy.deepcopy(_finite_relation_audit())
    mutate(audit)
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_atlas_finite_relation_index_annotations(audit_path, data)


def test_optional_handle_panel_uses_intrinsic_guard_phase_coordinates():
    plot = PlotHybridMorseSets(
        _FakeAtlasMorseGraph(),
        atlas_charts=_charts(),
        show_handles=True,
        handle_axis_labels=(r"$v_G$", r"$s$"),
        show_morse_graph=False,
        base_view="domain",
    )
    try:
        assert len(plot.figure.axes) == 2
        assert plot.handle_axis is not None
        assert plot.handle_axis.get_xlabel() == r"$v_G$"
        assert plot.handle_axis.get_ylabel() == r"$s$"
        assert plot.handle_axis.get_xlim() == pytest.approx((-1.0, 1.0))
        assert plot.handle_axis.get_ylim() == pytest.approx((0.0, 1.0))
        assert len(plot.handle_axis.collections) == 1
    finally:
        plt.close(plot.figure)


def test_selected_nodes_preserve_order_paths_through_hidden_morse_nodes():
    data = extract_atlas_morse_plot_data(_FakeAtlasMorseGraph(), _charts())
    order = atlas_morse_hasse(data, selected=(2, 0))
    assert set(order.nodes) == {0, 2}
    assert set(order.edges) == {(2, 0)}


def test_cached_data_needs_no_cmgdb_object_and_saves_figure(tmp_path):
    data = extract_atlas_morse_plot_data(_FakeAtlasMorseGraph(), _charts())
    plot = PlotHybridMorseSets(data, morse_nodes=(0,), show_morse_graph=False)
    try:
        outputs = save_hybrid_morse_figure(
            plot,
            tmp_path / "native-atlas",
            formats=("pdf", "png"),
        )
        assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    finally:
        plt.close(plot.figure)


def test_atlas_path_refuses_unsupported_or_unsubstantiated_overlays():
    graph = _FakeAtlasMorseGraph()
    with pytest.raises(ValueError, match="Conley-index labels are unavailable"):
        PlotHybridMorseSets(
            graph,
            atlas_charts=_charts(),
            conley_indices={0: ("x-1",)},
        )
    with pytest.raises(ValueError, match="show_transient is unavailable"):
        PlotHybridMorseSets(graph, atlas_charts=_charts(), show_transient=True)
    with pytest.raises(ValueError, match="legacy schematic"):
        PlotHybridMorseSets(
            graph,
            atlas_charts=_charts(),
            show_reset_attachments=True,
        )


def test_acceptance_wrapper_and_malformed_chart_boxes_are_checked():
    wrapper = SimpleNamespace(morse_graph=_FakeAtlasMorseGraph())
    data = extract_atlas_morse_plot_data(wrapper, _charts())
    assert data.vertex_ids == (0, 1, 2)

    malformed = _FakeAtlasMorseGraph()
    malformed._boxes[0] = ((9, (0.0, 0.0, 1.0, 1.0)),)
    with pytest.raises(ValueError, match="unknown Atlas chart 9"):
        extract_atlas_morse_plot_data(malformed, _charts())


def test_native_cmgdb_atlas_morse_graph_plots_without_untagging_boxes():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "AtlasModel"):
        pytest.skip("installed CMGDB predates AtlasModel")

    model = cmgdb.AtlasModel(1)
    model.add_chart(0, [-1.0, -1.0], [1.0, 1.0])
    model.add_chart(1, [-1.0, 0.0], [1.0, 1.0])
    model.set_map(lambda chart_id, bounds: [(chart_id, bounds)])
    morse_graph, _map_graph = cmgdb.ComputeMorseGraph(model)

    plot = PlotHybridMorseSets(morse_graph, atlas_charts=_charts())
    try:
        assert sum(len(component.boxes) for component in plot.components) > 0
        assert {
            box.chart_id
            for component in plot.components
            for box in component.boxes
        } == {0, 1}
    finally:
        plt.close(plot.figure)


def _small_set_data():
    """A large set, a thin strip on the left edge, and specks in a ring."""

    from hybrid_dynamics.src.atlas_morse_plot import (
        AtlasMorseBox,
        AtlasMorseNode,
        AtlasMorsePlotData,
    )

    large = (
        AtlasMorseBox(0, (0.5, -0.5, 1.5, 0.5)),
        AtlasMorseBox(1, (0.2, 0.0, 0.6, 1.0)),
    )
    strip = tuple(
        AtlasMorseBox(0, (x, y, x + 0.002, y + 0.002))
        for x in (0.0, 0.002)
        for y in np.linspace(-0.1, 0.098, 100)
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    specks = tuple(
        AtlasMorseBox(0, (x, y, x + 0.002, y + 0.002))
        for x, y in zip(1.8 + 0.1 * np.cos(angles), 0.7 + 0.1 * np.sin(angles))
    )
    return AtlasMorsePlotData(
        base_chart_id=0,
        handle_chart_id=1,
        base_bounds=((0.0, 2.0), (-1.0, 1.0)),
        handle_bounds=((0.0, 1.0), (0.0, 1.0)),
        nodes=(
            AtlasMorseNode(0, tuple(sorted(large))),
            AtlasMorseNode(1, strip),
            AtlasMorseNode(2, tuple(sorted(specks))),
        ),
        edges=((1, 0), (2, 0)),
        metadata={},
    )


def test_bin_coverage_is_the_covered_area_fraction_of_each_bin():
    from hybrid_dynamics.src.atlas_morse_plot import _bin_coverage

    bounds = np.array([[0.0, 0.0, 0.5, 0.25], [0.6, 0.3, 0.7, 0.4], [0.0, 0.5, 1.0, 1.0]])
    coverage = _bin_coverage(bounds, (0.0, 1.0), (0.0, 1.0), 4)
    assert coverage.sum() == pytest.approx(16 * (0.125 + 0.01 + 0.5))
    assert coverage[:2, 0] == pytest.approx([1.0, 1.0])
    assert coverage[2, 1] == pytest.approx(0.16)
    assert coverage[:, 2:] == pytest.approx(np.ones((4, 2)))
    # A cell across a bin corner splits its area among four bins.
    corner = _bin_coverage(np.array([[0.2, 0.2, 0.3, 0.3]]), (0.0, 1.0), (0.0, 1.0), 4)
    assert corner[:2, :2] == pytest.approx(np.full((2, 2), 0.04))


def test_frame_margin_keeps_boundary_cells_inside_the_axis_lines():
    from hybrid_dynamics.src.atlas_morse_plot import plot_atlas_hybrid_morse_sets

    data = _small_set_data()
    plot = plot_atlas_hybrid_morse_sets(
        data,
        clist=("#1f77b4", "#e6550d", "#31a354"),
        show_handles=True,
        show_morse_graph=False,
        base_view="domain",
        frame_margin=0.02,
    )
    try:
        base_axis = plot.projection_axes[0]
        assert base_axis.get_xlim() == pytest.approx((-0.04, 2.04))
        assert base_axis.get_ylim() == pytest.approx((-1.04, 1.04))
        assert plot.handle_axis.get_ylim() == pytest.approx((-0.02, 1.02))
        cells = min(collection.get_zorder() for collection in base_axis.collections)
        assert all(spine.get_zorder() < cells for spine in base_axis.spines.values())
        assert plot.zooms == () and plot.zoom_axes == ()
    finally:
        plt.close(plot.figure)


def test_detail_zooms_enlarge_the_sets_too_small_to_see():
    from hybrid_dynamics.src.atlas_morse_plot import plot_atlas_hybrid_morse_sets

    data = _small_set_data()
    plot = plot_atlas_hybrid_morse_sets(
        data,
        clist=("#1f77b4", "#e6550d", "#31a354"),
        show_handles=True,
        base_view="domain",
        handle_view="domain",
        frame_margin=0.02,
        detail_zooms=True,
    )
    try:
        # The strip on the left edge and the specks get one zoom each, left
        # to right; the large set and the handle chart get none.
        assert [(zoom.label, zoom.chart, zoom.morse_nodes) for zoom in plot.zooms] == [
            ("A", "base", (1,)),
            ("B", "base", (2,)),
        ]
        assert len(plot.zoom_axes) == 2
        for zoom, node in zip(plot.zooms, (1, 2)):
            boxes = data.nodes[node].boxes
            assert zoom.x_limits[0] <= min(box.lower[0] for box in boxes)
            assert zoom.x_limits[1] >= max(box.upper[0] for box in boxes)
            assert zoom.y_limits[0] <= min(box.lower[1] for box in boxes)
            assert zoom.y_limits[1] >= max(box.upper[1] for box in boxes)
            x_span = zoom.x_limits[1] - zoom.x_limits[0]
            y_span = zoom.y_limits[1] - zoom.y_limits[0]
            # Magnifications differ from the base panel's by at most 3.
            ratio = (x_span / 2.08) / (y_span / 2.08)
            assert 1.0 / 3.0 - 1e-9 <= ratio <= 3.0 + 1e-9
            assert zoom.to_dict()["morse_nodes"] == [node]
        # Windows outlined and labeled in the base panel.
        base_axis = plot.projection_axes[0]
        outlines = [patch for patch in base_axis.patches if not patch.get_fill()]
        assert len(outlines) == 2
        assert sorted(text.get_text() for text in base_axis.texts) == ["A", "B"]
        assert [axis.get_title(loc="left") for axis in plot.zoom_axes] == ["A", "B"]
        # The specks are still below the resolution of their zoom: marked.
        assert not plot.zoom_axes[0].lines
        (marks,) = plot.zoom_axes[1].lines
        assert len(marks.get_xdata()) == 8
        # The large set does not meet window A: zoom A draws the strip alone, opaque.
        assert all(
            collection.get_alpha() == pytest.approx(0.9)
            for collection in plot.zoom_axes[0].collections
        )
    finally:
        plt.close(plot.figure)
