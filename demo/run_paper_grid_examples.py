#!/usr/bin/env python3
"""Recompute the manuscript examples on the paper suspension grid ``Xi_n``.

For each example this runner builds ``Xi_n`` (``def:suspension-grid`` with the
cofiltration ``Xi_n = Xi_{n-1} ^ Xi_n(X_n)``), samples the multivalued map of
the Examples section, computes the Morse graph, attempts the finite-relation
Conley labels, and writes a figure (PDF and PNG) and a JSON summary to
``figures/paper_grid``.

The default image rule samples the four vertices of every elementary piece
of every atom (``--eval-mode corners``, the CMGDB default), takes the closed
atoms containing the endpoints, pads by one atom, and discards targets
outside the window.  ``--eval-mode center`` (padding forced) and
``--eval-mode random`` (``--num-pts``, ``--sample-depth``, ``--seed``) mirror
the other CMGDB evaluation modes; ``--eval-mode tensor`` with
``--samples-per-axis`` is the earlier tensor rule.  ``--gap-refinement-depth``
selects the opt-in gap refinement of :func:`compute_suspension_grid_relation`
(corners and tensor only).  Output names carry the
evaluation mode (``-corners``, ``-center``, ``-random10d4s0``, ``-tensor3``)
and, with gap refinement, the suffix ``-gap-refined``.  ``--tau EXAMPLE=T``
replaces the default ``tau`` of an example.  ``--level-offset EXAMPLE=K``
makes the base grid finer than the phase grid: level ``n`` then has
``2**(n + K)`` base cells per axis and phase cells of width ``2**(-n-2)``
(the cofiltration of the manuscript allows any contracting sequence of base
grids).  A positive offset adds ``-base<cells per axis>`` after the level in
the output names; offset ``0`` (the default) keeps the earlier names.  The
JSON summary records ``level_offset``, ``base_cells_per_axis``, and
``phase_cells``.  For the impacting van der Pol-Duffing oscillator the
summary also records which Morse sets contain points of the numerically known
invariant sets ``F``, ``Z``, ``C``, ``S``, ``U_Z``.

Besides the four examples, the positional names may be the parameter
variants of ``PAPER_GRID_VARIANTS`` (``impact-vdp-duffing-beta076``, the
oscillator at ``beta = 0.76``).  A variant is run only when named, its name
replaces the example's in the output names (so its outputs never overwrite
those of the example), and the summary records ``variant_of`` and the
replaced arguments ``variant_overrides``; ``parameters`` holds the values in
force.  ``--level``, ``--tau``, and ``--level-offset`` take a variant's name.

``--index-max-pieces N`` skips the index of a Morse set whose pair
``X = S cup F(S)`` has more than ``N`` elementary pieces (reported as blocked
with ``IndexSizeLimitError``); the quotient nerve of ``X`` is held in memory,
with about ten simplices per piece, so on fine base grids this bounds the time
and memory of the run.  There is no limit by default.  ``--index-workers N``
computes the indices of different Morse sets in ``N`` worker processes
(default 4), largest Morse set first; each worker holds its own copy of the
grid, about 1 GB at ``2**10`` and 4 GB at ``2**11`` base cells per axis.

The base samples are integrated many at a time
(:mod:`hybrid_dynamics.src.batched_suspension_flow`), with the step sequence
of ``solve_ivp``; ``--path-flow`` integrates each sample with
``SuspensionFlow`` instead, the reference.  The summary records which
(``integrator``).

``--figure-variants`` selects the figures: ``all`` (every Morse node, file
``<stem>.pdf``) and ``nontrivial`` (file ``<stem>-nontrivial.pdf``), which
hides the Morse nodes whose finite-relation index was computed and is trivial
(every homology dimension zero), keeps the nodes without a label, marked
with the dimensions of the relative homology of their pair, and draws the
order between the remaining nodes as reachability in
the full Morse graph, transitively reduced.  Both are written by default when
the index labels are computed.  The JSON summary records, per variant, the
shown and hidden nodes with the reason, and stores the atoms of every Morse
set, so ``demo/replot_paper_grid.py`` can redraw the figures from it.

Run from the ``code`` directory, for example::

    .venv/bin/python demo/run_paper_grid_examples.py --workers 12
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from hybrid_dynamics.examples.paper_grid_examples import (  # noqa: E402
    PAPER_GRID_PROBLEMS,
    PAPER_GRID_REFERENCE_SETS,
    paper_grid_example,
    paper_grid_names,
    paper_grid_overrides,
    paper_grid_problem,
    paper_grid_problem_factory,
)
from hybrid_dynamics.examples.paper_grid_figures import (  # noqa: E402
    write_paper_grid_figures,
)
from hybrid_dynamics.src.suspension_grid import (  # noqa: E402
    build_suspension_grid,
    check_suspension_grid,
)
from hybrid_dynamics.src.suspension_grid_conley import (  # noqa: E402
    compute_suspension_grid_conley_indices,
)
from hybrid_dynamics.src.suspension_grid_plot import (  # noqa: E402
    FIGURE_VARIANTS,
    RANGE_ENCODING,
    encode_index_ranges,
)
from hybrid_dynamics.src.suspension_grid_relation import (  # noqa: E402
    DEFAULT_EVAL_MODE,
    DEFAULT_NUM_PTS,
    DEFAULT_SAMPLE_DEPTH,
    DEFAULT_SEED,
    EVAL_MODES,
    EndpointCache,
    atom_set_components,
    audit_suspension_grid_endpoints,
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
    relation_image_connectivity,
)


DEFAULT_LEVELS = {
    "bouncing-ball": 5,
    "rimless-wheel": 6,
    "spiking-neuron": 8,
    "impact-vdp-duffing": 7,
    "impact-vdp-duffing-beta076": 7,
}


@functools.lru_cache(maxsize=1)
def _git_commit() -> dict[str, object]:
    """The commit and whether tracked files differ from it, read once per process.

    :func:`main` reads it before any output is written, so a run that
    overwrites tracked figures of an earlier run does not record its own
    outputs as uncommitted changes.
    """

    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=CODE_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    # Untracked files (such as this run's own outputs) do not make the tree dirty.
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(CODE_ROOT))
    except ValueError:
        return str(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "examples",
        nargs="*",
        choices=paper_grid_names(),
        help="examples or variants to run (default: the four examples, no variant)",
    )
    parser.add_argument(
        "--level",
        action="append",
        default=[],
        metavar="EXAMPLE=N",
        help="grid level per example (defaults: ball 5, wheel 6, neuron 8, impact 7)",
    )
    parser.add_argument(
        "--tau",
        action="append",
        default=[],
        metavar="EXAMPLE=T",
        help="suspension time per example (defaults: those of the example factories)",
    )
    parser.add_argument(
        "--level-offset",
        action="append",
        default=[],
        metavar="EXAMPLE=K",
        help=(
            "base grid offset per example: level N has 2**(N+K) base cells per axis "
            "and 2**(N+2) phase cells (default K=0)"
        ),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--path-flow",
        action="store_true",
        help=(
            "integrate each base sample with SuspensionFlow (scipy solve_ivp), the "
            "reference, instead of the batched integrator"
        ),
    )
    parser.add_argument(
        "--eval-mode",
        choices=EVAL_MODES,
        default=DEFAULT_EVAL_MODE,
        help="where each elementary piece is sampled (default: corners)",
    )
    parser.add_argument(
        "--num-pts",
        type=int,
        default=None,
        help=f"random mode: offsets per piece (default {DEFAULT_NUM_PTS})",
    )
    parser.add_argument(
        "--sample-depth",
        type=int,
        default=None,
        help=f"random mode: dyadic depth of the offsets (default {DEFAULT_SAMPLE_DEPTH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"random mode: seed of the offsets (default {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--samples-per-axis",
        type=int,
        default=None,
        help="tensor mode: samples per axis of each piece (default 3)",
    )
    parser.add_argument("--gap-refinement-depth", type=int, default=0)
    parser.add_argument("--no-conley", action="store_true")
    parser.add_argument(
        "--index-workers",
        type=int,
        default=4,
        metavar="N",
        help=(
            "worker processes for the indices of the Morse sets, largest first "
            "(default 4; each holds a copy of the grid)"
        ),
    )
    parser.add_argument(
        "--index-max-pieces",
        type=int,
        default=None,
        metavar="N",
        help=(
            "do not attempt the index of a Morse set whose pair X = S cup F(S) has more "
            "than N elementary pieces; it is reported as blocked (default: no limit)"
        ),
    )
    parser.add_argument(
        "--figure-variants",
        default=None,
        metavar="VARIANT[,VARIANT]",
        help=(
            "comma-separated figure variants from "
            f"{', '.join(FIGURE_VARIANTS)} (default: all,nontrivial; all with --no-conley)"
        ),
    )
    parser.add_argument("--probe-cells", type=int, default=150)
    parser.add_argument("--probe-intervals", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=CODE_ROOT / "figures" / "paper_grid"
    )
    arguments = parser.parse_args()
    random_options = {
        "--num-pts": arguments.num_pts,
        "--sample-depth": arguments.sample_depth,
        "--seed": arguments.seed,
    }
    if arguments.eval_mode != "random":
        given = [name for name, value in random_options.items() if value is not None]
        if given:
            parser.error(f"{', '.join(given)} requires --eval-mode random")
    if arguments.samples_per_axis is not None and arguments.eval_mode != "tensor":
        parser.error("--samples-per-axis requires --eval-mode tensor")
    if arguments.gap_refinement_depth > 0 and arguments.eval_mode in {"center", "random"}:
        parser.error("--gap-refinement-depth requires --eval-mode corners or tensor")
    if arguments.figure_variants is None:
        arguments.figure_variants = ("all",) if arguments.no_conley else FIGURE_VARIANTS
    else:
        variants = tuple(
            dict.fromkeys(
                value.strip() for value in arguments.figure_variants.split(",") if value.strip()
            )
        )
        unknown = [value for value in variants if value not in FIGURE_VARIANTS]
        if not variants or unknown:
            parser.error(
                f"--figure-variants takes values from {', '.join(FIGURE_VARIANTS)}; "
                f"got {arguments.figure_variants!r}"
            )
        if "nontrivial" in variants and arguments.no_conley:
            parser.error("--figure-variants nontrivial needs the index labels (drop --no-conley)")
        arguments.figure_variants = variants
    return arguments


def _sampling_options(arguments: argparse.Namespace) -> dict[str, object]:
    """Keyword arguments of :func:`compute_suspension_grid_relation`."""

    options: dict[str, object] = {"eval_mode": arguments.eval_mode}
    if arguments.eval_mode == "random":
        options.update(
            num_pts=DEFAULT_NUM_PTS if arguments.num_pts is None else arguments.num_pts,
            sample_depth=(
                DEFAULT_SAMPLE_DEPTH
                if arguments.sample_depth is None
                else arguments.sample_depth
            ),
            seed=DEFAULT_SEED if arguments.seed is None else arguments.seed,
        )
    if arguments.eval_mode == "tensor":
        options["samples_per_axis"] = (
            3 if arguments.samples_per_axis is None else arguments.samples_per_axis
        )
    return options


def _sampling_suffix(options: dict[str, object]) -> str:
    mode = options["eval_mode"]
    if mode == "random":
        return f"-random{options['num_pts']}d{options['sample_depth']}s{options['seed']}"
    if mode == "tensor":
        return f"-tensor{options['samples_per_axis']}"
    return f"-{mode}"


def _identify_nodes(
    grid, morse, reference_sets: dict[str, dict[str, np.ndarray]]
) -> dict[str, object]:
    """Which Morse sets contain points of the named invariant sets.

    A point counts for a Morse set if a closed piece of one of its atoms
    contains the point (base points on an identification count on both
    sides).  Per set: the number of points, the count per Morse node, and
    the count lying in no Morse set.
    """

    node_of_atom = np.full(grid.n_atoms, -1, dtype=np.int64)
    for index, morse_set in enumerate(morse.morse_sets):
        node_of_atom[morse_set] = index
    result: dict[str, object] = {}
    for name, points in reference_sets.items():
        base = np.asarray(points["base"], dtype=float).reshape(-1, 2)
        handle = np.asarray(points["handle"], dtype=float).reshape(-1, 2)
        located_base = grid.locate_base_points(base)
        located_handle = grid.locate_handle_points(handle[:, 0], handle[:, 1])
        rows = np.concatenate(
            (located_base.point_index, base.shape[0] + located_handle.point_index)
        )
        nodes = node_of_atom[grid.atom_of_piece[
            np.concatenate((located_base.piece, located_handle.piece))
        ]]
        total = base.shape[0] + handle.shape[0]
        per_node = {}
        for node in np.unique(nodes[nodes >= 0]).tolist():
            per_node[str(node)] = int(np.unique(rows[nodes == node]).size)
        in_some = np.unique(rows[nodes >= 0])
        result[name] = {
            "points": int(total),
            "base_points": int(base.shape[0]),
            "handle_points": int(handle.shape[0]),
            "points_per_node": per_node,
            "points_in_no_morse_set": int(total - in_some.size),
        }
    return result


def _grid_suffix(level: int, level_offset: int) -> str:
    """``-level<n>``, followed by ``-base<cells per axis>`` for a positive offset."""

    text = f"-level{level}"
    if level_offset:
        text += f"-base{2 ** (level + level_offset)}"
    return text


def _output_stem(
    name: str, tau: float, level: int, level_offset: int, sampling_suffix: str
) -> str:
    """``paper-grid-<example or variant>-tau<100 tau>-level<n>[-base<m>]<sampling>``."""

    return (
        f"paper-grid-{name}-tau{int(round(tau * 100)):03d}"
        f"{_grid_suffix(level, level_offset)}{sampling_suffix}"
    )


def _run(
    name: str,
    level: int,
    arguments: argparse.Namespace,
    tau: float | None = None,
    level_offset: int = 0,
) -> dict[str, object]:
    timings: dict[str, float] = {}
    options: dict[str, object] = {} if tau is None else {"tau": float(tau)}
    if level_offset:
        options["level_offset"] = int(level_offset)
    problem = paper_grid_problem(name, **options)
    factory = paper_grid_problem_factory(name, **options)
    depth = int(arguments.gap_refinement_depth)
    sampling = _sampling_options(arguments)
    suffix = _sampling_suffix(sampling) + ("-gap-refined" if depth > 0 else "")
    stem = _output_stem(name, problem.tau, level, problem.window.level_offset, suffix)
    print(
        f"== {name}: parameters {problem.parameters}; level {level}, base cells per axis "
        f"{problem.window.cells_per_axis(level)}, phase cells {2 ** (level + 2)}, "
        f"tau {problem.tau}, sampling {sampling}, gap refinement {depth}",
        flush=True,
    )

    started = time.perf_counter()
    grid = build_suspension_grid(problem.window, problem.guard, level)
    check = check_suspension_grid(grid)
    timings["grid"] = time.perf_counter() - started
    print(f"grid: {grid.summary()} check={check.passed}", flush=True)

    # Endpoints are evaluated once and reused by the path exit policy below.
    endpoint_cache = EndpointCache()
    started = time.perf_counter()
    relation = compute_suspension_grid_relation(
        grid,
        problem,
        **sampling,
        gap_refinement_depth=depth,
        workers=arguments.workers,
        problem_factory=factory,
        progress=lambda message: print(message, flush=True),
        endpoint_cache=endpoint_cache,
        batched_flow=not arguments.path_flow,
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
            **sampling,
            exit_policy="path",
            gap_refinement_depth=depth,
            workers=arguments.workers,
            problem_factory=factory,
            endpoint_cache=endpoint_cache,
            batched_flow=not arguments.path_flow,
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
        del path_relation, path_morse
    else:
        path_policy = {"identical_to_endpoint_policy": True}
    print(
        f"endpoint cache: {endpoint_cache.evaluated} samples evaluated, "
        f"{endpoint_cache.reused} reused",
        flush=True,
    )
    del endpoint_cache

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

    reference_sets = PAPER_GRID_REFERENCE_SETS.get(name)
    identification = (
        _identify_nodes(grid, morse, reference_sets(problem)) if reference_sets else None
    )
    if identification is not None:
        print(f"node identification: {identification}", flush=True)

    conley: list[dict[str, object]] = []
    if not arguments.no_conley:
        started = time.perf_counter()
        results = compute_suspension_grid_conley_indices(
            relation,
            morse.morse_sets,
            workers=arguments.index_workers,
            problem_factory=factory,
            max_pieces=arguments.index_max_pieces,
        )
        for index, result in enumerate(results):
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
    started = time.perf_counter()
    figures = write_paper_grid_figures(
        grid,
        morse.morse_sets,
        morse.edges,
        conley,
        example=name,
        tau=problem.tau,
        level=level,
        output_stem=output_dir / stem,
        audit_path=summary_path,
        variants=arguments.figure_variants,
        display_path=_display_path,
    )
    timings["figure"] = time.perf_counter() - started
    for variant, record in figures["figure_variants"].items():
        print(
            f"figure {variant}: {len(record['shown_nodes'])} nodes shown, "
            f"{len(record['hidden_nodes'])} hidden, "
            f"{len(record['blocked_nodes_shown'])} blocked; {record['files']}",
            flush=True,
        )

    summary = {
        "schema": "paper-suspension-grid-run-v3",
        "example": name,
        "variant_of": None if paper_grid_example(name) == name else paper_grid_example(name),
        "variant_overrides": paper_grid_overrides(name),
        "parameters": problem.parameters,
        "tau": problem.tau,
        "level": level,
        "level_offset": problem.window.level_offset,
        "base_cells_per_axis": grid.cells_per_axis,
        "phase_cells": grid.n_phase,
        "a_n": grid.a_n,
        "construction": (
            "Xi_n generated by K_j(mu) and Q_j(mu,k) for j <= n (def:suspension-grid, "
            "Xi_n = Xi_{n-1} ^ Xi_n(X_n)); elementary pieces with generator signatures"
        ),
        "image_rule": {
            "description": relation.statistics["image_rule"],
            "default_rule": relation.statistics["default_rule"],
            "eval_mode": relation.statistics["eval_mode"],
            "evaluation_offsets": relation.statistics["evaluation_offsets"],
            "offset_denominator": relation.statistics["offset_denominator"],
            "samples_per_piece": relation.statistics["samples_per_piece"],
            "num_pts": relation.statistics["num_pts"],
            "sample_depth": relation.statistics["sample_depth"],
            "seed": relation.statistics["seed"],
            "samples_per_axis": relation.statistics["samples_per_axis"],
            "padding": relation.statistics["padding"],
            "padding_forced": relation.statistics["padding_forced"],
            "gap_refinement_depth": depth,
            "exit_policy": relation.statistics["exit_policy"],
            "command_line": sys.argv[1:],
        },
        "integrator": (
            "SuspensionFlow paths (scipy solve_ivp, RK45), one per sample"
            if arguments.path_flow or problem.batch_dynamics is None
            else (
                "batched RK45 (batched_suspension_flow) with the step sequence of "
                "solve_ivp; handle samples by SuspensionFlow paths"
            )
        ),
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
            "morse_set_atoms": [encode_index_ranges(values) for values in morse.morse_sets],
            "morse_set_atoms_encoding": RANGE_ENCODING,
        },
        "morse_graph_path_exit_policy": path_policy,
        "reference_set_identification": identification,
        "conley": conley,
        "conley_options": {
            "max_pieces": arguments.index_max_pieces,
            "index_workers": arguments.index_workers,
        },
        "conley_labels_in_figure": {str(key): list(value) for key, value in labels.items()},
        "continuous_system_conley_index_certified": False,
        "discarded_exits": {
            "endpoints": relation.statistics["discarded_exit_endpoints"],
            "source_atoms": relation.statistics["source_atoms_with_exit"],
        },
        "runtime_seconds": {**timings, "relation_detail": {
            key: value for key, value in relation.statistics.items() if key.startswith("seconds")
        }},
        "figures": figures["figures"],
        "figure_variants": figures["figure_variants"],
        "code": dict(_git_commit()),
        "python": platform.python_version(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)
    return summary


def main() -> int:
    arguments = _arguments()
    _git_commit()
    levels = dict(DEFAULT_LEVELS)
    for entry in arguments.level:
        key, value = entry.split("=", 1)
        if key not in levels:
            raise SystemExit(f"unknown example in --level: {key}")
        levels[key] = int(value)
    taus: dict[str, float] = {}
    for entry in arguments.tau:
        key, value = entry.split("=", 1)
        if key not in paper_grid_names():
            raise SystemExit(f"unknown example in --tau: {key}")
        taus[key] = float(value)
    offsets: dict[str, int] = {}
    for entry in arguments.level_offset:
        key, value = entry.split("=", 1)
        if key not in paper_grid_names():
            raise SystemExit(f"unknown example in --level-offset: {key}")
        if int(value) < 0:
            raise SystemExit(f"--level-offset must be nonnegative: {entry}")
        offsets[key] = int(value)
    for name in arguments.examples or list(PAPER_GRID_PROBLEMS):
        _run(name, levels[name], arguments, taus.get(name), offsets.get(name, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
