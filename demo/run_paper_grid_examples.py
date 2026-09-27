#!/usr/bin/env python3
"""Recompute the manuscript examples on the paper suspension grid ``Xi_n``.

For each example this runner builds ``Xi_n`` (``def:suspension-grid`` with the
cofiltration ``Xi_n = Xi_{n-1} ^ Xi_n(X_n)``), samples the multivalued map of
the Examples section (``3 x 3`` samples per piece, the closed atoms containing
the endpoints, one-atom padding, targets outside the window discarded),
computes the Morse graph, attempts the finite-relation Conley labels, and
writes a figure (PDF and PNG) and a JSON summary to ``figures/paper_grid``.

``--gap-refinement-depth`` selects the opt-in gap refinement of
:func:`compute_suspension_grid_relation`, which is not part of the
manuscript's description; its outputs carry the suffix ``-gap-refined``.

Run from the ``code`` directory, for example::

    .venv/bin/python demo/run_paper_grid_examples.py --workers 12
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from hybrid_dynamics import (  # noqa: E402
    PlotHybridMorseSets,
    save_hybrid_morse_figure,
)
from hybrid_dynamics.examples.paper_grid_examples import (  # noqa: E402
    PAPER_GRID_PROBLEMS,
    paper_grid_problem_factory,
)
from hybrid_dynamics.src.atlas_morse_plot import (  # noqa: E402
    AtlasFiniteRelationIndexAnnotations,
)
from hybrid_dynamics.src.suspension_grid import (  # noqa: E402
    build_suspension_grid,
    check_suspension_grid,
)
from hybrid_dynamics.src.suspension_grid_conley import (  # noqa: E402
    compute_suspension_grid_conley_index,
)
from hybrid_dynamics.src.suspension_grid_plot import (  # noqa: E402
    suspension_grid_morse_plot_data,
)
from hybrid_dynamics.src.suspension_grid_relation import (  # noqa: E402
    atom_set_components,
    audit_suspension_grid_endpoints,
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
    relation_image_connectivity,
)


DEFAULT_LEVELS = {"bouncing-ball": 5, "rimless-wheel": 6, "spiking-neuron": 8}
FIGURE_STYLE = {
    "bouncing-ball": {"labels": (r"$h$", r"$v$"), "handle": (r"$v_G$", r"$s$"), "view": "domain"},
    "rimless-wheel": {
        "labels": (r"$\theta$", r"$\dot\theta$"),
        "handle": (r"$\dot\theta_G$", r"$s$"),
        "view": "support",
    },
    "spiking-neuron": {"labels": (r"$v$", r"$u$"), "handle": (r"$u_G$", r"$s$"), "view": "support"},
}


def _git_commit() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=CODE_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(CODE_ROOT))
    except ValueError:
        return str(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", nargs="*", choices=tuple(PAPER_GRID_PROBLEMS))
    parser.add_argument(
        "--level",
        action="append",
        default=[],
        metavar="EXAMPLE=N",
        help="grid level per example (defaults: ball 5, wheel 6, neuron 8)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gap-refinement-depth", type=int, default=0)
    parser.add_argument("--no-conley", action="store_true")
    parser.add_argument("--probe-cells", type=int, default=150)
    parser.add_argument("--probe-intervals", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=CODE_ROOT / "figures" / "paper_grid"
    )
    return parser.parse_args()


def _run(name: str, level: int, arguments: argparse.Namespace) -> dict[str, object]:
    timings: dict[str, float] = {}
    problem = PAPER_GRID_PROBLEMS[name]()
    factory = paper_grid_problem_factory(name)
    depth = int(arguments.gap_refinement_depth)
    suffix = "-gap-refined" if depth > 0 else ""
    stem = f"paper-grid-{name}-tau{int(round(problem.tau * 100)):03d}-level{level}{suffix}"
    print(f"== {name}: level {level}, tau {problem.tau}, gap refinement {depth}", flush=True)

    started = time.perf_counter()
    grid = build_suspension_grid(problem.window, problem.guard, level)
    check = check_suspension_grid(grid)
    timings["grid"] = time.perf_counter() - started
    print(f"grid: {grid.summary()} check={check.passed}", flush=True)

    started = time.perf_counter()
    relation = compute_suspension_grid_relation(
        grid,
        problem,
        gap_refinement_depth=depth,
        workers=arguments.workers,
        problem_factory=factory,
        progress=lambda message: print(message, flush=True),
    )
    timings["relation"] = time.perf_counter() - started

    started = time.perf_counter()
    morse = compute_suspension_morse_graph(relation)
    timings["morse_graph"] = time.perf_counter() - started
    print(
        f"relation edges {relation.n_edges}; Morse sets "
        f"{[len(values) for values in morse.morse_sets]}; edges {morse.edges}",
        flush=True,
    )

    # Alternative reading of the exit rule: discard endpoints whose path left D.
    path_policy: dict[str, object] = {}
    if relation.statistics["left_and_returned_endpoints"]:
        started = time.perf_counter()
        path_relation = compute_suspension_grid_relation(
            grid,
            problem,
            exit_policy="path",
            gap_refinement_depth=depth,
            workers=arguments.workers,
            problem_factory=factory,
        )
        path_morse = compute_suspension_morse_graph(path_relation)
        path_policy = {
            "edges": path_relation.n_edges,
            "morse_set_atoms": [int(values.size) for values in path_morse.morse_sets],
            "morse_set_base_cells": [
                int(values.size) for values in path_morse.base_readouts(grid)
            ],
            "morse_edges": [list(edge) for edge in path_morse.edges],
            "seconds": time.perf_counter() - started,
        }
        print(f"path exit policy: {path_policy}", flush=True)
    else:
        path_policy = {"identical_to_endpoint_policy": True}

    started = time.perf_counter()
    connectivity = relation_image_connectivity(relation)
    timings["image_connectivity"] = time.perf_counter() - started
    print(f"image connectivity: {connectivity}", flush=True)

    started = time.perf_counter()
    rng = np.random.default_rng(2026)
    probe_cells = rng.choice(grid.n_base, size=min(arguments.probe_cells, grid.n_base), replace=False)
    probe_intervals = rng.choice(
        grid.n_guard, size=min(arguments.probe_intervals, grid.n_guard), replace=False
    )
    audit = audit_suspension_grid_endpoints(
        relation,
        problem,
        base_cells=probe_cells,
        subdivision=4,
        guard_intervals=probe_intervals,
        seed=2026,
    )
    timings["endpoint_probe"] = time.perf_counter() - started
    probe = {
        "base_cells_probed": int(probe_cells.size),
        "base_probe_lattice": "5x5 per cell",
        "guard_intervals_probed": int(probe_intervals.size),
        "phase_probes_per_interval": 7,
        "witnesses_in_window": len(audit.witnesses),
        "missed": len(audit.missed),
        "evaluation_failures": len(audit.evaluation_failures),
    }
    print(f"endpoint probe: {probe}", flush=True)

    nodes = []
    for index, morse_set in enumerate(morse.morse_sets):
        pieces = np.concatenate([grid.atom(int(atom)) for atom in morse_set])
        base_cells = grid.base_readout(morse_set)
        _guard, phases = grid.handle_indices(pieces[pieces >= grid.n_base])
        nodes.append(
            {
                "index": index,
                "atoms": int(morse_set.size),
                "pieces": int(pieces.size),
                "base_readout_cells": int(base_cells.size),
                "handle_pieces": int(np.count_nonzero(pieces >= grid.n_base)),
                "handle_phase_intervals_met": int(np.unique(phases).size),
                "components_in_suspension": atom_set_components(grid, morse_set),
                "base_readout_bounds": {
                    "lower": grid.base_bounds(base_cells)[:, :2].min(axis=0).tolist()
                    if base_cells.size
                    else None,
                    "upper": grid.base_bounds(base_cells)[:, 2:].max(axis=0).tolist()
                    if base_cells.size
                    else None,
                },
            }
        )

    conley: list[dict[str, object]] = []
    if not arguments.no_conley:
        started = time.perf_counter()
        for index, morse_set in enumerate(morse.morse_sets):
            result = compute_suspension_grid_conley_index(relation, morse_set, morse_node=index)
            conley.append(result.to_dict())
            print(f"Conley M({index}): {result.to_dict()}", flush=True)
        timings["conley"] = time.perf_counter() - started

    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{stem}.json"
    labels = {
        int(entry["morse_node"]): tuple(entry["shift_class"])
        for entry in conley
        if entry["computed"]
    }
    plot_data = suspension_grid_morse_plot_data(
        grid,
        morse,
        metadata={"model": name, "t_star": problem.tau, "level": level},
    )
    annotations = (
        AtlasFiniteRelationIndexAnnotations(
            shift_classes=labels,
            coefficient_field=5,
            result_scope="finite_reset_quotient_relation",
            audit_path=summary_path,
        )
        if labels
        else None
    )
    style = FIGURE_STYLE[name]
    started = time.perf_counter()
    plot = PlotHybridMorseSets(
        plot_data,
        finite_relation_annotations=annotations,
        axis_labels=style["labels"],
        handle_axis_labels=style["handle"],
        show_handles=False,
        show_morse_graph=True,
        show_legend=False,
        show_panel_titles=False,
        show_component_sizes=False,
        show_status_note=False,
        base_view=style["view"],
        fig_h=3.4,
    )
    try:
        outputs = save_hybrid_morse_figure(plot, output_dir / stem, formats=("pdf", "png"), dpi=400)
    finally:
        plt.close(plot.figure)
    timings["figure"] = time.perf_counter() - started

    summary = {
        "schema": "paper-suspension-grid-run-v1",
        "example": name,
        "parameters": problem.parameters,
        "tau": problem.tau,
        "level": level,
        "level_offset": problem.window.level_offset,
        "a_n": grid.a_n,
        "construction": (
            "Xi_n generated by K_j(mu) and Q_j(mu,k) for j <= n (def:suspension-grid, "
            "Xi_n = Xi_{n-1} ^ Xi_n(X_n)); elementary pieces with generator signatures"
        ),
        "image_rule": {
            "samples_per_axis": relation.statistics["samples_per_axis"],
            "tensor_rule": "corners, edge midpoints, and center of every elementary piece",
            "padding": relation.statistics["padding"],
            "gap_refinement_depth": depth,
            "in_manuscript": depth == 0,
            "exit_policy": relation.statistics["exit_policy"],
        },
        "grid": grid.summary(),
        "grid_check": {
            "passed": check.passed,
            **{key: bool(value) for key, value in vars(check).items()},
        },
        "relation": {
            key: value
            for key, value in relation.statistics.items()
            if not key.startswith("seconds")
        },
        "image_connectivity": connectivity,
        "endpoint_probe": probe,
        "morse_graph": {
            "nodes": nodes,
            "edges": [list(edge) for edge in morse.edges],
            "scc_count": morse.n_components,
        },
        "morse_graph_path_exit_policy": path_policy,
        "conley": conley,
        "conley_labels_in_figure": {str(key): list(value) for key, value in labels.items()},
        "continuous_system_conley_index_certified": False,
        "discarded_exits": {
            "endpoints": relation.statistics["discarded_exit_endpoints"],
            "source_atoms": relation.statistics["source_atoms_with_exit"],
        },
        "runtime_seconds": {**timings, "relation_detail": {
            key: value for key, value in relation.statistics.items() if key.startswith("seconds")
        }},
        "figures": [_display_path(path) for path in outputs],
        "code": _git_commit(),
        "python": platform.python_version(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return summary


def main() -> int:
    arguments = _arguments()
    levels = dict(DEFAULT_LEVELS)
    for entry in arguments.level:
        key, value = entry.split("=", 1)
        if key not in levels:
            raise SystemExit(f"unknown example in --level: {key}")
        levels[key] = int(value)
    for name in arguments.examples or list(PAPER_GRID_PROBLEMS):
        _run(name, levels[name], arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
