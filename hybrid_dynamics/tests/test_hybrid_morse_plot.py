"""Focused tests for suspension-aware Morse-set plotting."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import pytest
from matplotlib import patches

from hybrid_dynamics.src.hybrid_morse_plot import (
    CMGDB_MORSE_PALETTE,
    PlotHybridMorseSets,
    hybrid_morse_components,
    hybrid_morse_hasse,
    plot_hybrid_morse_sets,
    save_hybrid_morse_figure,
)
from hybrid_dynamics.src.sampled_suspension import BaseCell, PhaseCell


@dataclass(frozen=True)
class _CemeteryCell:
    label: str


def _small_result():
    base_zero = BaseCell(0)
    base_one = BaseCell(1)
    transient = BaseCell(2)
    phase_zero = PhaseCell(guard_cell=1, slab=0)
    phase_one = PhaseCell(guard_cell=1, slab=1)
    cemetery = _CemeteryCell("failure")
    graph = nx.DiGraph()
    graph.add_edges_from(
        (
            (base_zero, base_one),
            (base_one, phase_zero),
            (phase_zero, phase_one),
            (phase_one, base_zero),
            (transient, base_zero),
            (cemetery, cemetery),
        ),
    )
    handle = SimpleNamespace(
        handle_id="reset-1",
        guard_cell=base_one,
        reset_cells=(base_zero,),
        phase_cells=(phase_zero, phase_one),
        guard_state=(0.75, -0.25, 0.10, -0.15),
        reset_state=(-0.75, -0.10, -0.10, -0.05),
    )
    ingredients = SimpleNamespace(
        base_bounds=((-1.0, 1.0), (-0.5, 0.5), (-0.2, 0.2), (-0.3, 0.1)),
        base_subdivisions=(3, 2, 2, 2),
        handles=(handle,),
        failure_cells=(cemetery,),
    )
    return SimpleNamespace(
        model_name="garcia_passive_walker_period_two_suspension",
        t_star=0.5,
        handle_slabs=2,
        relation_graph=graph,
        recurrent_sccs=(
            frozenset((base_zero, base_one, phase_zero, phase_one)),
            frozenset((cemetery,)),
        ),
        suspension_complex_ingredients=ingredients,
    )


def test_component_colors_follow_cmgdb_indices_and_classify_cell_kinds():
    result = _small_result()
    components = hybrid_morse_components(result)

    assert [component.label for component in components] == ["M(0)", "M(1)"]
    assert components[0].color == CMGDB_MORSE_PALETTE[0]
    assert components[1].color == "#8f8f8f"
    assert len(components[0].base_cells) == 2
    assert len(components[0].phase_cells) == 2
    assert not components[0].is_cemetery
    assert components[1].is_cemetery

    # Filtering retains the original component index and therefore its color.
    assert hybrid_morse_components(result, morse_nodes=(1,))[0].color == "#8f8f8f"


def test_plot_separates_base_projections_handles_and_cemetery_marker():
    plot = PlotHybridMorseSets(
        _small_result(),
        show_transient=True,
        show_cemetery=True,
        show_reset_attachments=True,
    )
    try:
        assert plot.projections == ((0, 1), (2, 3))
        assert len(plot.projection_axes) == 2
        assert plot.handle_axis is not None
        assert plot.morse_graph_axis is not None
        assert len(plot.figure.axes) == 4

        # Each base panel has recurrent boxes, light transient support, and a
        # schematic reset attachment.  Handle phase is instead in its own lane.
        assert all(len(axis.patches) >= 3 for axis in plot.projection_axes)
        assert len(plot.handle_axis.patches) == 2
        assert len(plot.handle_axis.collections) >= 3
        assert plot.handle_axis.get_title() == ""
        assert any(text.get_text() == "cemetery" for text in plot.handle_axis.texts)
        assert plot.projection_axes[0].get_xlabel() == r"$\theta$"
        assert plot.projection_axes[1].get_ylabel() == r"$\dot\phi$"
        assert set(plot.morse_graph.nodes) == {0, 1}
        assert not plot.morse_graph.edges
        assert {text.get_text() for text in plot.morse_graph_axis.texts} == {
            "M(0)",
            "M(1)",
        }
    finally:
        plt.close(plot.figure)


def test_snake_case_alias_and_paper_output_helper(tmp_path):
    plot = plot_hybrid_morse_sets(
        _small_result(),
        proj_dims=(0, 1),
        show_transient=False,
    )
    try:
        outputs = save_hybrid_morse_figure(
            plot,
            tmp_path / "hybrid_morse",
            formats=("pdf", "png", "svg"),
        )
        assert [path.suffix for path in outputs] == [".pdf", ".png", ".svg"]
        assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    finally:
        plt.close(plot.figure)


def test_default_paper_view_omits_auxiliary_cemetery_component():
    plot = PlotHybridMorseSets(_small_result(), proj_dims=(0, 1))
    try:
        assert [component.index for component in plot.components] == [0]
        assert set(plot.morse_graph.nodes) == {0}
        assert {text.get_text() for text in plot.morse_graph_axis.texts} == {"M(0)"}
    finally:
        plt.close(plot.figure)


def test_no_handle_layout_propagates_explicit_conley_index_label():
    plot = PlotHybridMorseSets(
        _small_result(),
        proj_dims=(0, 1),
        show_handles=False,
        conley_indices={0: ("x-1", "0")},
    )
    try:
        assert plot.handle_axis is None
        assert len(plot.projection_axes) == 1
        assert plot.morse_graph_axis is not None
        assert len(plot.figure.axes) == 2
        assert {text.get_text() for text in plot.morse_graph_axis.texts} == {
            "M(0)\n(x-1, 0)",
        }
        assert any(
            isinstance(patch, patches.Ellipse)
            for patch in plot.morse_graph_axis.patches
        )
    finally:
        plt.close(plot.figure)


def test_projection_and_palette_validation():
    result = _small_result()
    with pytest.raises(IndexError):
        PlotHybridMorseSets(result, proj_dims=(0, 4))
    with pytest.raises(ValueError):
        hybrid_morse_components(result, palette=())

    result.recurrent_sccs = result.recurrent_sccs[:1]
    with pytest.raises(ValueError, match="recurrent_sccs"):
        PlotHybridMorseSets(result)


def test_morse_order_uses_paths_through_transient_sccs():
    source = BaseCell(0)
    transient = BaseCell(1)
    target = BaseCell(2)
    graph = nx.DiGraph(
        (
            (source, source),
            (source, transient),
            (transient, target),
            (target, target),
        ),
    )
    result = _small_result()
    result.relation_graph = graph
    result.recurrent_sccs = (frozenset((source,)), frozenset((target,)))
    result.suspension_complex_ingredients = SimpleNamespace(
        base_bounds=((-1.0, 1.0), (-1.0, 1.0)),
        base_subdivisions=(3, 1),
        handles=(),
        failure_cells=(),
    )

    order = hybrid_morse_hasse(result)
    assert set(order.edges) == {(0, 1)}
