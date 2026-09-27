"""Figure variants of the paper-grid runs and the replot from a JSON summary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import patches  # noqa: E402

from hybrid_dynamics import PlotHybridMorseSets, build_suspension_grid  # noqa: E402
from hybrid_dynamics.examples.paper_grid_examples import (  # noqa: E402
    bouncing_ball_problem,
    paper_grid_problem,
)
from hybrid_dynamics.examples.paper_grid_figures import (  # noqa: E402
    COLOR_RULE,
    FRAME_MARGIN,
    PALETTE_GRAYS,
    PANEL_MAX_PIXELS,
    PAPER_GRID_PALETTE,
    PAPER_GRID_PALETTE_NAME,
    TRIVIAL_INDEX_COLOR,
    _panel_dpi,
    draw_paper_grid_figure,
    draw_paper_grid_panels,
    paper_grid_color_record,
    paper_grid_morse_colors,
    write_paper_grid_figures,
)
from hybrid_dynamics.src.atlas_morse_plot import (  # noqa: E402
    CELL_OUTLINE_WIDTH,
    PANEL_CHART_SIZE,
    PANEL_GRAPH_FONT_SIZE,
    PANEL_HANDLE_SIZE,
    PANEL_ZOOM_SIZE,
    ZOOM_FIGURE_STYLE,
    AtlasFiniteRelationIndexAnnotations,
    atlas_morse_components,
)
from hybrid_dynamics.src.hybrid_morse_plot import (  # noqa: E402
    CMGDB_MORSE_PALETTE,
    MORSE_LABEL_DARK,
    MORSE_LABEL_LIGHT,
    MorseNodeLabel,
    _morse_node_size,
    morse_graph_node_labels,
)
from hybrid_dynamics.src.suspension_grid_plot import (  # noqa: E402
    TRIVIAL_INDEX_REASON,
    decode_index_ranges,
    encode_index_ranges,
    index_status,
    morse_figure_selection,
    restricted_morse_order,
    suspension_grid_morse_sets_plot_data,
)


CODE_ROOT = Path(__file__).resolve().parents[2]

# Hasse diagram of six Morse nodes (node 0 is the attractor).  Node 1 and
# node 3 have a trivial index, node 2 a blocked one.
EDGES = ((1, 0), (2, 1), (3, 0), (4, 2), (4, 3), (5, 4))
STATUS = {0: "nontrivial", 1: "trivial", 2: "blocked", 3: "trivial", 4: "nontrivial", 5: "nontrivial"}


def _record(node: int, status: str) -> dict[str, object]:
    if status == "blocked":
        return {
            "morse_node": node,
            "computed": False,
            "shift_class": [],
            "homology_dimensions": [],
            "blocker": "ValueError: carrier image is not acyclic over GF(5)",
        }
    nontrivial = status == "nontrivial"
    return {
        "morse_node": node,
        "computed": True,
        "shift_class": ["x-1" if nontrivial else "0", "0", "0"],
        "homology_dimensions": [1 if nontrivial else 0, 0, 0],
        "blocker": "",
    }


CONLEY = [_record(node, status) for node, status in STATUS.items()]


def test_nontrivial_variant_hides_trivial_nodes_and_keeps_blocked_ones():
    assert [index_status(entry) for entry in CONLEY] == list(STATUS.values())

    selection = morse_figure_selection(6, EDGES, CONLEY, "nontrivial")

    assert selection.shown == (0, 2, 4, 5)
    assert selection.hidden == ((1, TRIVIAL_INDEX_REASON), (3, TRIVIAL_INDEX_REASON))
    assert selection.blocked == (2,)
    # 2 reaches 0 only through the hidden node 1; 4 -> 0 is implied by
    # 4 -> 2 -> 0 and is removed by the transitive reduction.
    assert selection.order == ((2, 0), (4, 2), (5, 4))
    record = selection.to_dict()
    assert [entry["morse_node"] for entry in record["hidden_nodes"]] == [1, 3]
    assert record["blocked_nodes_shown"] == [2]
    assert record["order_shown"] == [[2, 0], [4, 2], [5, 4]]

    every = morse_figure_selection(6, EDGES, CONLEY, "all")
    assert every.shown == tuple(range(6)) and every.hidden == ()
    assert every.blocked == (2,)
    assert every.order == tuple(sorted(EDGES))
    # Without index records only the full figure can be drawn.
    assert morse_figure_selection(6, EDGES, [], "all").blocked == ()
    with pytest.raises(ValueError, match="index record for every"):
        morse_figure_selection(6, EDGES, CONLEY[:5], "nontrivial")
    with pytest.raises(ValueError, match="unknown figure variant"):
        morse_figure_selection(6, EDGES, CONLEY, "labeled")
    assert restricted_morse_order(3, ((2, 1), (1, 0)), (0, 2)) == ((2, 0),)


def test_index_ranges_round_trip():
    values = np.array([7, 0, 1, 2, 5, 8, 12, 2])
    text = encode_index_ranges(values)
    assert text == "0-2 5 7-8 12"
    assert decode_index_ranges(text).tolist() == [0, 1, 2, 5, 7, 8, 12]
    assert encode_index_ranges([]) == "" and decode_index_ranges("").size == 0
    with pytest.raises(ValueError):
        decode_index_ranges("4-2")


def _ball_run(level: int = 2):
    problem = bouncing_ball_problem()
    grid = build_suspension_grid(problem.window, problem.guard, level)
    atoms = np.linspace(0, grid.n_atoms - 1, 6).astype(np.int64)
    morse_sets = [np.array([atom]) for atom in atoms]
    return problem, grid, morse_sets


def test_blocked_nodes_are_marked_and_hidden_nodes_keep_colors():
    _problem, grid, morse_sets = _ball_run()
    data = suspension_grid_morse_sets_plot_data(grid, morse_sets, EDGES)
    selection = morse_figure_selection(6, EDGES, CONLEY, "nontrivial")
    labels = {0: ("x-1", "0", "0"), 4: ("x-1", "0", "0"), 5: ("x-1", "0", "0")}
    annotations = AtlasFiniteRelationIndexAnnotations(
        shift_classes=labels,
        coefficient_field=5,
        result_scope="finite_reset_quotient_relation",
        audit_path=Path("run.json"),
    )
    full = PlotHybridMorseSets(data, finite_relation_annotations=annotations)
    plot = PlotHybridMorseSets(
        data,
        morse_nodes=selection.shown,
        finite_relation_annotations=annotations,
        blocked_index_nodes=selection.blocked,
    )
    try:
        colors = {component.index: component.color for component in full.components}
        assert {component.index: component.color for component in plot.components} == {
            node: colors[node] for node in selection.shown
        }
        assert tuple(sorted(plot.morse_graph.edges)) == selection.order
        texts = sorted(morse_graph_node_labels(plot.morse_graph_axis))
        assert texts == ["M(0)\n(x-1, 0, 0)", "M(2)\nblocked", "M(4)\n(x-1, 0, 0)", "M(5)\n(x-1, 0, 0)"]
        ellipses = [
            patch for patch in plot.morse_graph_axis.patches if isinstance(patch, patches.Ellipse)
        ]
        assert sum(patch.get_linestyle() != "solid" for patch in ellipses) == 1
    finally:
        plt.close(full.figure)
        plt.close(plot.figure)
    with pytest.raises(ValueError, match="index label and be blocked"):
        PlotHybridMorseSets(
            data, finite_relation_annotations=annotations, blocked_index_nodes=(0,)
        )
    plt.close("all")


def test_paper_figure_draws_the_base_chart_and_enlarges_boundary_cells():
    problem = bouncing_ball_problem(tau=0.5, level_offset=5)
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    # The atoms of the handle pieces over the first guard interval: their base
    # cells are two cells on the edge h = 0, at v = -5 and v = 4.
    handle = np.unique(grid.atom_of_piece[grid.handle_piece(0, np.arange(grid.n_phase))])
    base_cells = grid.base_bounds(grid.base_readout(handle))
    assert len(base_cells) == 2 and np.all(base_cells[:, 0] == 0.0)
    data = suspension_grid_morse_sets_plot_data(grid, [handle], [])
    plot = draw_paper_grid_figure(data, example="bouncing-ball", shown=(0,))
    try:
        (base_axis,) = plot.projection_axes
        # The set has base cells, so the figure has no handle chart.
        assert plot.handle_axes == ()
        (h_lower, h_upper), _v_bounds = grid.window.ambient_bounds
        margin = FRAME_MARGIN * (h_upper - h_lower)
        assert base_axis.get_xlim() == pytest.approx((h_lower - margin, h_upper + margin))
        # The two base cells are specks too far apart for a zoom: drawn at
        # their true extent and outlined in the color of the set, no symbol.
        assert plot.zooms == ()
        assert not base_axis.lines
        fill, outline = base_axis.collections
        for collection in (fill, outline):
            drawn = sorted(
                tuple(np.r_[path.vertices.min(axis=0), path.vertices.max(axis=0)])
                for path in collection.get_paths()
            )
            assert drawn == pytest.approx(sorted(map(tuple, base_cells)))
        assert not np.any(fill.get_linewidths())
        assert outline.get_linewidths() == pytest.approx([CELL_OUTLINE_WIDTH])
        blended = [
            1.0 - 0.9 * (1.0 - value)
            for value in matplotlib.colors.to_rgb(plot.components[0].color)
        ]
        assert outline.get_edgecolor()[0] == pytest.approx([*blended, 1.0])
    finally:
        plt.close(plot.figure)

    # The base cell at the origin alone gets a zoom next to the base panel.
    lower_corners = grid.base_bounds(np.arange(grid.n_base))[:, :2]
    origin = int(grid.d_map[int(np.argmin(np.abs(lower_corners).sum(axis=1)))])
    data = suspension_grid_morse_sets_plot_data(grid, [np.array([origin])], [])
    plot = draw_paper_grid_figure(data, example="bouncing-ball", shown=(0,))
    try:
        (zoom,) = plot.zooms
        assert (zoom.label, zoom.chart, zoom.morse_nodes) == ("A", "base", (0,))
        assert zoom.x_limits[0] < 0.0 < zoom.x_limits[1]
        assert plot.zoom_axes[0].get_title(loc="left") == "A"
        # The cell is outlined in the zoom as in the panel, with no symbol.
        (zoom_axis,) = plot.zoom_axes
        assert not zoom_axis.lines and not plot.projection_axes[0].lines
        assert [
            list(collection.get_linewidths()) for collection in zoom_axis.collections
        ] == [[0.0], [CELL_OUTLINE_WIDTH]]
    finally:
        plt.close(plot.figure)


def _combined_names(figures) -> list[str]:
    """Names of the combined figure files of each variant, without the panels."""

    return [
        Path(path).name
        for record in figures["figure_variants"].values()
        for path in record["files"]
    ]


def _load_replot_script():
    path = CODE_ROOT / "demo" / "replot_paper_grid.py"
    spec = importlib.util.spec_from_file_location("replot_paper_grid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_figures_and_replot_from_the_json(tmp_path):
    problem, grid, morse_sets = _ball_run()
    summary_path = tmp_path / "paper-grid-bouncing-ball-tau150-level2-corners.json"
    figures = write_paper_grid_figures(
        grid,
        morse_sets,
        EDGES,
        CONLEY,
        example="bouncing-ball",
        tau=problem.tau,
        level=2,
        output_stem=summary_path.with_suffix(""),
        audit_path=summary_path,
        variants=("all", "nontrivial"),
        dpi=60,
    )
    assert _combined_names(figures) == [
        "paper-grid-bouncing-ball-tau150-level2-corners.pdf",
        "paper-grid-bouncing-ball-tau150-level2-corners.png",
        "paper-grid-bouncing-ball-tau150-level2-corners-nontrivial.pdf",
        "paper-grid-bouncing-ball-tau150-level2-corners-nontrivial.png",
    ]
    assert all(Path(path).is_file() for path in figures["figures"])
    nontrivial = figures["figure_variants"]["nontrivial"]
    assert nontrivial["shown_nodes"] == [0, 2, 4, 5]
    assert all(
        isinstance(record["zooms"], list) and "marked_in_panel" not in record
        for record in figures["figure_variants"].values()
    )
    assert [entry["morse_node"] for entry in nontrivial["hidden_nodes"]] == [1, 3]
    # Each variant records the palette and the color of every Morse node, the
    # same in both variants.
    for record in figures["figure_variants"].values():
        assert record["colors"] == paper_grid_color_record(6, CONLEY)
        assert record["colors"]["palette"] == PAPER_GRID_PALETTE_NAME == "CMGDB default"
        assert record["colors"]["palette_colors"] == list(PAPER_GRID_PALETTE)
        assert record["colors"]["trivial_index_color"] == TRIVIAL_INDEX_COLOR
        assert record["colors"]["rule"] == COLOR_RULE
        assert record["colors"]["morse_node_colors"] == [
            {"morse_node": node, "color": color}
            for node, color in paper_grid_morse_colors(6, CONLEY).items()
        ]
        assert record["colors"]["palette_cycled"] is False
        assert "palette_cycled_note" not in record["colors"]

    # A JSON summary with the fields the runner writes; its figure records are
    # in the older form that listed the sets marked by squares in a panel and
    # had no panel figures or colors.
    summary = {
        "schema": "paper-suspension-grid-run-v3",
        "example": "bouncing-ball",
        "tau": problem.tau,
        "level": 2,
        "grid": json.loads(json.dumps(grid.summary())),
        "morse_graph": {
            "nodes": [{"index": index, "atoms": 1} for index in range(6)],
            "edges": [list(edge) for edge in EDGES],
            "morse_set_atoms": [encode_index_ranges(values) for values in morse_sets],
        },
        "conley": CONLEY,
        "figures": [
            path for record in figures["figure_variants"].values() for path in record["files"]
        ],
        "figure_variants": {
            variant: {
                **{
                    key: value
                    for key, value in record.items()
                    if key not in {"panel_files", "colors"}
                },
                "marked_in_panel": [{"chart": "base", "morse_node": 0}],
            }
            for variant, record in figures["figure_variants"].items()
        },
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for path in figures["figures"]:
        Path(path).unlink()

    replot = _load_replot_script()
    result = replot.replot(summary_path)
    assert result["figure_variants"] == figures["figure_variants"]
    assert result["figures"] == figures["figures"]
    assert all(Path(path).is_file() for path in result["figures"])
    updated = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated["figure_variants"]["nontrivial"]["hidden_nodes"] == nontrivial["hidden_nodes"]
    assert all(
        "marked_in_panel" not in record for record in updated["figure_variants"].values()
    )
    assert updated["figure_variants"]["all"]["panel_files"] == figures["figure_variants"][
        "all"
    ]["panel_files"]
    assert all(
        record["colors"] == paper_grid_color_record(6, CONLEY)
        for record in updated["figure_variants"].values()
    )
    assert updated["figures"] == figures["figures"]
    assert updated["figures_replotted"]["script"] == "demo/replot_paper_grid.py"

    older = dict(
        summary,
        morse_graph={
            key: value for key, value in summary["morse_graph"].items() if key != "morse_set_atoms"
        },
    )
    older_path = tmp_path / "older.json"
    older_path.write_text(json.dumps(older), encoding="utf-8")
    with pytest.raises(replot.ReplotError, match="morse_set_atoms"):
        replot.replot(older_path)
    other_level = dict(summary, level=3)
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other_level), encoding="utf-8")
    with pytest.raises(replot.ReplotError, match="rebuilt grid"):
        replot.replot(other_path)


def _load_runner_script():
    path = CODE_ROOT / "demo" / "run_paper_grid_examples.py"
    spec = importlib.util.spec_from_file_location("run_paper_grid_examples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_names_record_a_finer_base_grid():
    runner = _load_runner_script()
    assert runner._grid_suffix(6, 0) == "-level6"
    assert runner._grid_suffix(6, 4) == "-level6-base1024"
    assert runner._grid_suffix(7, 1) == "-level7-base256"


def test_a_variant_has_its_own_output_names_and_replots(tmp_path):
    runner = _load_runner_script()
    names = [
        runner._output_stem(name, 1.0, 6, 4, "-corners-gap-refined")
        for name in ("impact-vdp-duffing", "impact-vdp-duffing-beta076")
    ]
    assert names == [
        "paper-grid-impact-vdp-duffing-tau100-level6-base1024-corners-gap-refined",
        "paper-grid-impact-vdp-duffing-beta076-tau100-level6-base1024-corners-gap-refined",
    ]
    problem = paper_grid_problem("impact-vdp-duffing-beta076", tau=1.0, level_offset=1)
    grid = build_suspension_grid(problem.window, problem.guard, 1)
    morse_sets = [np.array([int(grid.d_map[0])])]
    summary = {
        "schema": "paper-suspension-grid-run-v3",
        "example": "impact-vdp-duffing-beta076",
        "variant_of": "impact-vdp-duffing",
        "variant_overrides": {"beta": 0.76},
        "tau": 1.0,
        "level": 1,
        "level_offset": 1,
        "grid": json.loads(json.dumps(grid.summary())),
        "morse_graph": {
            "nodes": [{"index": 0, "atoms": 1}],
            "edges": [],
            "morse_set_atoms": [encode_index_ranges(values) for values in morse_sets],
        },
        "conley": [_record(0, "nontrivial")],
    }
    summary_path = tmp_path / "paper-grid-impact-vdp-duffing-beta076-tau100-level1-base4-corners.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    result = _load_replot_script().replot(summary_path, variants=("all",))
    assert _combined_names(result) == [
        "paper-grid-impact-vdp-duffing-beta076-tau100-level1-base4-corners.pdf",
        "paper-grid-impact-vdp-duffing-beta076-tau100-level1-base4-corners.png",
    ]


def test_replot_rebuilds_the_base_offset_of_the_run(tmp_path):
    problem = bouncing_ball_problem(tau=1.5, level_offset=1)
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    assert grid.cells_per_axis == 8 and grid.n_phase == 16
    morse_sets = [np.array([int(grid.d_map[0])]), np.array([int(grid.d_map[1])])]
    conley = [_record(0, "nontrivial"), _record(1, "trivial")]
    summary = {
        "schema": "paper-suspension-grid-run-v3",
        "example": "bouncing-ball",
        "tau": problem.tau,
        "level": 2,
        "level_offset": 1,
        "base_cells_per_axis": 8,
        "phase_cells": 16,
        "grid": json.loads(json.dumps(grid.summary())),
        "morse_graph": {
            "nodes": [{"index": 0, "atoms": 1}, {"index": 1, "atoms": 1}],
            "edges": [[1, 0]],
            "morse_set_atoms": [encode_index_ranges(values) for values in morse_sets],
        },
        "conley": conley,
    }
    summary_path = tmp_path / "paper-grid-bouncing-ball-tau150-level2-base8-corners.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    replot = _load_replot_script()
    result = replot.replot(summary_path)
    assert _combined_names(result) == [
        "paper-grid-bouncing-ball-tau150-level2-base8-corners.pdf",
        "paper-grid-bouncing-ball-tau150-level2-base8-corners.png",
        "paper-grid-bouncing-ball-tau150-level2-base8-corners-nontrivial.pdf",
        "paper-grid-bouncing-ball-tau150-level2-base8-corners-nontrivial.png",
    ]
    assert result["figure_variants"]["nontrivial"]["shown_nodes"] == [0]
    # Without the offset the rebuilt grid has 4 cells per axis, not the recorded 8.
    without_offset = {key: value for key, value in summary.items() if key != "level_offset"}
    other_path = tmp_path / "without-offset.json"
    other_path.write_text(json.dumps(without_offset), encoding="utf-8")
    with pytest.raises(replot.ReplotError, match="rebuilt grid"):
        replot.replot(other_path)


def test_a_node_without_a_label_shows_its_homology_dimensions():
    from hybrid_dynamics.examples.paper_grid_figures import blocked_index_line

    with_homology = {"homology_computed": True, "homology_dimensions": [0, 1, 1]}
    assert blocked_index_line(with_homology) == "dim H (0, 1, 1)"
    assert blocked_index_line({"homology_computed": False, "homology_dimensions": []}) == "blocked"
    assert blocked_index_line(_record(2, "blocked")) == "blocked"

    _problem, grid, morse_sets = _ball_run()
    data = suspension_grid_morse_sets_plot_data(grid, morse_sets, EDGES)
    plot = PlotHybridMorseSets(data, blocked_index_nodes={2: "dim H (0, 1, 1)"})
    try:
        assert "M(2)\ndim H (0, 1, 1)" in morse_graph_node_labels(plot.morse_graph_axis)
        dashed = [
            patch
            for patch in plot.morse_graph_axis.patches
            if isinstance(patch, patches.Ellipse) and patch.get_linestyle() != "solid"
        ]
        assert len(dashed) == 1
    finally:
        plt.close(plot.figure)


def test_paper_figure_draws_the_handle_chart_only_for_a_set_without_base_cells():
    problem = bouncing_ball_problem(tau=0.5, level_offset=5)
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    # Atoms of the middle phase pieces over the first guard interval: no base
    # cell reads out to them, so the base chart alone would not show the set.
    middle = np.arange(grid.n_phase // 4, 3 * grid.n_phase // 4)
    atoms = np.unique(grid.atom_of_piece[grid.handle_piece(0, middle)])
    assert grid.base_readout(atoms).size == 0
    data = suspension_grid_morse_sets_plot_data(grid, [atoms], [])
    plot = draw_paper_grid_figure(data, example="bouncing-ball", shown=(0,))
    try:
        (handle_axis,) = plot.handle_axes
        assert handle_axis.get_xlabel() == r"$v_G$"
        assert handle_axis.get_ylabel() == r"$s$"
        assert handle_axis.get_ylim() == pytest.approx((-FRAME_MARGIN, 1.0 + FRAME_MARGIN))
        pieces = np.concatenate([grid.atom(int(atom)) for atom in atoms])
        cells = handle_axis.collections[0]
        assert len(cells.get_paths()) == np.count_nonzero(pieces >= grid.n_base)
        assert tuple(cells.get_facecolor()[0][:3]) == pytest.approx(
            matplotlib.colors.to_rgb(plot.components[0].color)
        )
    finally:
        plt.close(plot.figure)


def _zoom_and_handle_run():
    """A ball run with a Morse set drawn in a zoom and one with no base cell.

    ``M(0)`` is the base cell at the origin, too small to see in the base
    chart; ``M(1)`` is the atoms of the middle phase pieces over the first
    guard interval, drawn only in the handle chart.
    """

    problem = bouncing_ball_problem(tau=0.5, level_offset=5)
    grid = build_suspension_grid(problem.window, problem.guard, 2)
    lower_corners = grid.base_bounds(np.arange(grid.n_base))[:, :2]
    origin = int(grid.d_map[int(np.argmin(np.abs(lower_corners).sum(axis=1)))])
    middle = np.arange(grid.n_phase // 4, 3 * grid.n_phase // 4)
    handle = np.unique(grid.atom_of_piece[grid.handle_piece(0, middle)])
    return problem, grid, [np.array([origin]), handle]


def _cells(axis) -> list[tuple[object, ...]]:
    """Per collection of an axis: its rectangles, line widths, and colors."""

    return [
        (
            sorted(
                tuple(np.round(np.r_[path.vertices.min(axis=0), path.vertices.max(axis=0)], 12))
                for path in collection.get_paths()
            ),
            tuple(collection.get_linewidths()),
            tuple(map(tuple, collection.get_facecolor())),
            tuple(map(tuple, collection.get_edgecolor())),
        )
        for collection in axis.collections
    ]


def _axes_inches(axis) -> tuple[float, float]:
    position = axis.get_position()
    width, height = axis.figure.get_size_inches()
    return (position.width * width, position.height * height)


def test_each_panel_is_written_and_recorded_for_both_variants(tmp_path):
    problem, grid, morse_sets = _zoom_and_handle_run()
    # M(1), the set drawn in the handle chart, has a trivial index, so the
    # nontrivial variant has no handle chart.
    conley = [_record(0, "nontrivial"), _record(1, "trivial")]
    stem = tmp_path / "paper-grid-bouncing-ball-tau050-level2-corners"
    figures = write_paper_grid_figures(
        grid,
        morse_sets,
        [(1, 0)],
        conley,
        example="bouncing-ball",
        tau=problem.tau,
        level=2,
        output_stem=stem,
        audit_path=stem.with_suffix(".json"),
        variants=("all", "nontrivial"),
        dpi=60,
    )
    expected = {
        "all": (stem.name, ["base", "zoom-A", "handle", "graph"]),
        "nontrivial": (f"{stem.name}-nontrivial", ["base", "zoom-A", "graph"]),
    }
    listed: list[str] = []
    for variant, (variant_stem, panels) in expected.items():
        record = figures["figure_variants"][variant]
        assert [Path(path).name for path in record["files"]] == [
            f"{variant_stem}.pdf",
            f"{variant_stem}.png",
        ]
        assert [zoom["label"] for zoom in record["zooms"]] == ["A"]
        assert list(record["panel_files"]) == panels
        for name, paths in record["panel_files"].items():
            assert [Path(path).name for path in paths] == [
                f"{variant_stem}-{name}.pdf",
                f"{variant_stem}-{name}.png",
            ]
            assert all(Path(path).is_file() for path in paths)
        listed += record["files"]
        listed += [path for paths in record["panel_files"].values() for path in paths]
    # The figures list has the files of each variant, then those of its panels.
    assert figures["figures"] == listed


def test_panel_figures_match_the_combined_figure():
    _problem, grid, morse_sets = _zoom_and_handle_run()
    data = suspension_grid_morse_sets_plot_data(grid, morse_sets, [(1, 0)])
    plot = draw_paper_grid_figure(data, example="bouncing-ball", shown=(0, 1))
    panels = draw_paper_grid_panels(data, example="bouncing-ball", shown=(0, 1))
    try:
        assert list(panels.figures) == ["base", "zoom-A", "handle", "graph"]
        (zoom,) = plot.zooms
        assert panels.zooms == plot.zooms
        # Each chart panel has the limits, labels, and cells of the combined
        # figure (with the same outline rule), at about the same size.
        for name, combined, size in (
            ("base", plot.projection_axes[0], PANEL_CHART_SIZE),
            ("handle", plot.handle_axes[0], PANEL_HANDLE_SIZE),
        ):
            axis = panels.axes[name]
            assert axis.get_xlim() == combined.get_xlim()
            assert axis.get_ylim() == combined.get_ylim()
            assert (axis.get_xlabel(), axis.get_ylabel()) == (
                combined.get_xlabel(),
                combined.get_ylabel(),
            )
            assert _cells(axis) == _cells(combined)
            assert _axes_inches(axis) == pytest.approx(size)
        # The base panel outlines the zoom window, with its letter, as the
        # combined figure does.
        for axis in (panels.axes["base"], plot.projection_axes[0]):
            (mark,) = [patch for patch in axis.patches if isinstance(patch, patches.Rectangle)]
            assert (
                mark.get_x(),
                mark.get_x() + mark.get_width(),
                mark.get_y(),
                mark.get_y() + mark.get_height(),
            ) == pytest.approx((*zoom.x_limits, *zoom.y_limits))
            assert [text.get_text() for text in axis.texts] == ["A"]
        # The zoom panel shows the same window and cells, with its letter, in
        # a square of PANEL_ZOOM_SIZE with larger tick labels.
        zoom_axis = panels.axes["zoom-A"]
        (combined_zoom,) = plot.zoom_axes
        assert zoom_axis.get_xlim() == combined_zoom.get_xlim() == pytest.approx(zoom.x_limits)
        assert zoom_axis.get_ylim() == combined_zoom.get_ylim() == pytest.approx(zoom.y_limits)
        assert zoom_axis.get_title(loc="left") == "A"
        assert _cells(zoom_axis) == _cells(combined_zoom)
        assert _axes_inches(zoom_axis) == pytest.approx(PANEL_ZOOM_SIZE)
        tick = zoom_axis.xaxis.get_major_ticks()[0]
        assert tick.label1.get_fontsize() == ZOOM_FIGURE_STYLE.tick_label_size
        # Shown apart from the base panel, the zoom names its coordinates as
        # that panel does; next to it in the combined figure, it does not.
        base = panels.axes["base"]
        assert (zoom_axis.get_xlabel(), zoom_axis.get_ylabel()) == (
            base.get_xlabel(),
            base.get_ylabel(),
        )
        assert zoom_axis.get_xlabel() and zoom_axis.get_ylabel()
        assert (combined_zoom.get_xlabel(), combined_zoom.get_ylabel()) == ("", "")
        assert sorted(morse_graph_node_labels(panels.axes["graph"])) == sorted(
            morse_graph_node_labels(plot.morse_graph_axis)
        )
    finally:
        plt.close(plot.figure)
        panels.close()


def test_the_graph_panel_is_sized_from_its_layout():
    _problem, grid, morse_sets = _ball_run()
    chain = [(node + 1, node) for node in range(5)]
    data = suspension_grid_morse_sets_plot_data(grid, morse_sets, chain)
    annotations = AtlasFiniteRelationIndexAnnotations(
        shift_classes={node: ("x-1", "0", "0") for node in range(6)},
        coefficient_field=5,
        result_scope="finite_reset_quotient_relation",
        audit_path=Path("run.json"),
    )
    sizes = []
    for shown in ((0, 1), tuple(range(6))):
        panels = draw_paper_grid_panels(
            data, example="bouncing-ball", shown=shown, annotations=annotations, dpi=100
        )
        try:
            figure, axis = panels.figures["graph"], panels.axes["graph"]
            # The figure is the layout, and one unit of the layout is one
            # inch, so the labels print at PANEL_GRAPH_FONT_SIZE points.
            layout = (np.diff(axis.get_xlim())[0], np.diff(axis.get_ylim())[0])
            assert tuple(figure.get_size_inches()) == pytest.approx(layout)
            origin, unit = axis.transData.transform([(0.0, 0.0), (1.0, 1.0)])
            assert unit - origin == pytest.approx([figure.dpi, figure.dpi])
            ellipses = [patch for patch in axis.patches if isinstance(patch, patches.Ellipse)]
            labels = [patch for patch in axis.patches if isinstance(patch, MorseNodeLabel)]
            assert len(ellipses) == len(labels) == len(shown)
            for label in labels:
                vertices = label.get_path().vertices
                center = vertices.mean(axis=0)
                ellipse = min(
                    ellipses,
                    key=lambda patch: float(np.hypot(*(np.asarray(patch.center) - center))),
                )
                assert (ellipse.width, ellipse.height) == pytest.approx(
                    _morse_node_size(label.text, PANEL_GRAPH_FONT_SIZE)
                )
                scaled = ((vertices[:, 0] - ellipse.center[0]) / (ellipse.width / 2.0)) ** 2 + (
                    (vertices[:, 1] - ellipse.center[1]) / (ellipse.height / 2.0)
                ) ** 2
                assert scaled.max() < 1.0, label.text
            sizes.append(tuple(figure.get_size_inches()))
        finally:
            panels.close()
    (small_width, small_height), (large_width, large_height) = sizes
    assert large_height > small_height + 1.0 and large_width >= small_width


def test_a_very_large_panel_png_has_a_lower_resolution():
    figure = plt.figure(figsize=(40.0, 10.0))
    try:
        # The side of the saved file includes the padding of the tight box.
        side = 40.0 + 2.0 * plt.rcParams["savefig.pad_inches"]
        dpi = _panel_dpi(figure, 400)
        assert dpi < 400 and dpi * side <= PANEL_MAX_PIXELS < (dpi + 1) * side
        figure.set_size_inches(3.0, 3.0)
        assert _panel_dpi(figure, 400) == 400
    finally:
        plt.close(figure)


def test_the_paper_grid_palette_is_cmgdb_and_trivial_sets_are_gray():
    assert PAPER_GRID_PALETTE == CMGDB_MORSE_PALETTE
    assert PAPER_GRID_PALETTE_NAME == "CMGDB default"
    assert TRIVIAL_INDEX_COLOR == "#BBBBBB" and TRIVIAL_INDEX_COLOR not in PAPER_GRID_PALETTE
    assert PALETTE_GRAYS <= {color.lower() for color in PAPER_GRID_PALETTE}

    # Nodes 1 and 3 have a trivial index and are gray; node i of the others
    # (a nontrivial label, or node 2 without a label) takes palette color i,
    # as in CMGDB.
    palette = PAPER_GRID_PALETTE
    assert paper_grid_morse_colors(6, CONLEY) == {
        0: palette[0],
        1: TRIVIAL_INDEX_COLOR,
        2: palette[2],
        3: TRIVIAL_INDEX_COLOR,
        4: palette[4],
        5: palette[5],
    }
    # Without index records every node takes its palette color.
    assert paper_grid_morse_colors(3) == {0: palette[0], 1: palette[1], 2: palette[2]}

    # A colored node whose palette color is gray takes the next color that is
    # not gray, so gray always means a trivial index.
    colors = paper_grid_morse_colors(24)
    assert palette[7].lower() in PALETTE_GRAYS and colors[7] == palette[8]
    assert palette[22].lower() in PALETTE_GRAYS and palette[23].lower() in PALETTE_GRAYS
    assert colors[22] == colors[23] == palette[24 % len(palette)]
    assert not {color.lower() for color in colors.values()} & PALETTE_GRAYS

    # A colored node numbered past the palette: the colors repeat, and the
    # record says so.
    record = paper_grid_color_record(len(palette) + 2)
    assert record["palette_cycled"] is True
    assert f"repeat from Morse node {len(palette)}" in record["palette_cycled_note"]
    assert record["morse_node_colors"][len(palette)]["color"] == palette[0]
    # A trivial node past the end of the palette is gray and does not cycle.
    conley = [_record(node, "nontrivial") for node in range(len(palette))] + [
        _record(len(palette), "trivial")
    ]
    record = paper_grid_color_record(len(palette) + 1, conley)
    assert record["palette_cycled"] is False and "palette_cycled_note" not in record

    # A palette given as a mapping must color every drawn node.
    _problem, grid, morse_sets = _ball_run()
    data = suspension_grid_morse_sets_plot_data(grid, morse_sets, EDGES)
    with pytest.raises(ValueError, match="no color for the Morse nodes"):
        atlas_morse_components(data, morse_nodes=(0, 1), palette={0: "#000000"})


def _fill_colors(axis) -> set[str]:
    """Colors of the cells drawn on a chart axis (not of their outlines)."""

    return {
        matplotlib.colors.to_hex(collection.get_facecolor()[0][:3])
        for collection in axis.collections
        if not np.any(collection.get_linewidths())
    }


def test_a_morse_set_has_one_color_in_every_variant_and_panel():
    _problem, grid, morse_sets = _ball_run()
    data = suspension_grid_morse_sets_plot_data(grid, morse_sets, EDGES)
    colors = paper_grid_morse_colors(6, CONLEY)
    # Label text is white on the dark indigo and green, near black elsewhere.
    label_colors = {
        0: MORSE_LABEL_LIGHT,
        1: MORSE_LABEL_DARK,
        2: MORSE_LABEL_DARK,
        3: MORSE_LABEL_DARK,
        4: MORSE_LABEL_DARK,
        5: MORSE_LABEL_LIGHT,
    }
    drawn_colors = {}
    for variant in ("all", "nontrivial"):
        selection = morse_figure_selection(6, EDGES, CONLEY, variant)
        options = {
            "example": "bouncing-ball",
            "shown": selection.shown,
            "blocked": selection.blocked,
            "colors": colors,
        }
        plot = draw_paper_grid_figure(data, **options)
        panels = draw_paper_grid_panels(data, **options, dpi=60)
        try:
            expected = {node: colors[node] for node in selection.shown}
            for drawing in (plot, panels):
                assert {part.index: part.color for part in drawing.components} == expected
            chart_axes = [*plot.projection_axes, *plot.handle_axes, *plot.zoom_axes] + [
                axis for name, axis in panels.axes.items() if name != "graph"
            ]
            allowed = {color.lower() for color in expected.values()}
            for axis in chart_axes:
                assert _fill_colors(axis) <= allowed
            for axis in (plot.morse_graph_axis, panels.axes["graph"]):
                ellipses = [patch for patch in axis.patches if isinstance(patch, patches.Ellipse)]
                labels = [patch for patch in axis.patches if isinstance(patch, MorseNodeLabel)]
                nodes = sorted(selection.shown)
                assert [matplotlib.colors.to_hex(patch.get_facecolor()) for patch in ellipses] == [
                    colors[node].lower() for node in nodes
                ]
                assert [matplotlib.colors.to_hex(patch.get_facecolor()) for patch in labels] == [
                    label_colors[node] for node in nodes
                ]
            drawn_colors[variant] = {
                part.index: part.color for part in [*plot.components, *panels.components]
            }
        finally:
            plt.close(plot.figure)
            panels.close()
    # The gray of the trivial sets is drawn only in the all variant, and the
    # other sets have the same color in both variants.
    assert {drawn_colors["all"][node] for node in (1, 3)} == {TRIVIAL_INDEX_COLOR}
    assert TRIVIAL_INDEX_COLOR not in drawn_colors["nontrivial"].values()
    assert drawn_colors["nontrivial"] == {
        node: color for node, color in drawn_colors["all"].items() if node not in (1, 3)
    }
