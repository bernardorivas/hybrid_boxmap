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
from hybrid_dynamics.examples.paper_grid_examples import bouncing_ball_problem  # noqa: E402
from hybrid_dynamics.examples.paper_grid_figures import write_paper_grid_figures  # noqa: E402
from hybrid_dynamics.src.atlas_morse_plot import (  # noqa: E402
    AtlasFiniteRelationIndexAnnotations,
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
        texts = sorted(text.get_text() for text in plot.morse_graph_axis.texts)
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
    assert [Path(path).name for path in figures["figures"]] == [
        "paper-grid-bouncing-ball-tau150-level2-corners.pdf",
        "paper-grid-bouncing-ball-tau150-level2-corners.png",
        "paper-grid-bouncing-ball-tau150-level2-corners-nontrivial.pdf",
        "paper-grid-bouncing-ball-tau150-level2-corners-nontrivial.png",
    ]
    assert all(Path(path).is_file() for path in figures["figures"])
    nontrivial = figures["figure_variants"]["nontrivial"]
    assert nontrivial["shown_nodes"] == [0, 2, 4, 5]
    assert [entry["morse_node"] for entry in nontrivial["hidden_nodes"]] == [1, 3]

    # A JSON summary with the fields the runner writes.
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
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for path in figures["figures"]:
        Path(path).unlink()

    replot = _load_replot_script()
    result = replot.replot(summary_path)
    assert result["figure_variants"] == {
        variant: {**record, "files": result["figure_variants"][variant]["files"]}
        for variant, record in figures["figure_variants"].items()
    }
    assert all(Path(path).is_file() for path in result["figures"])
    updated = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated["figure_variants"]["nontrivial"]["hidden_nodes"] == nontrivial["hidden_nodes"]
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
