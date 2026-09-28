#!/usr/bin/env python3
"""Compute the missing index labels of paper-grid runs from their JSON summaries.

For each JSON summary written by ``demo/run_paper_grid_examples.py`` this
script rebuilds the problem and the grid ``Xi_n`` of the run as
``demo/replot_paper_grid.py`` does (the rebuilt grid must match the ``grid``
record), and recomputes the relation with the image rule recorded under
``image_rule`` and the integrator recorded under ``integrator`` (summaries
without it predate the batched integrator and are recomputed with
``SuspensionFlow`` paths).  The summary is refused, and nothing is changed,
unless the recomputed relation has the recorded image rule and number of
edges (``relation.edges``) and its Morse sets (``morse_graph.morse_set_atoms``,
in node order) and Morse graph (``morse_graph.edges``) equal the recorded
ones.

The index of the image pair is then computed, with ``index_map="auto"``
(the exit components, with the excision pair as fallback), only for the
Morse nodes whose record has no label (``computed`` false).  The recomputed
pair must have the recorded numbers of atoms and pieces (and, when both
were computed, the recorded homology dimensions); otherwise the summary is
refused.  Only the records of those nodes are replaced, by the new records
(``index_map: "auto"``, ``label_source``, ``exit_components_blocker``, see
``archive/notes/PAPER_GRID.md``); every other record is kept as it was, and so is
``conley_options``, which describes the run.  ``conley_labels_in_figure``
gains the new labels, and ``labels_filled`` (a list, one entry per
application) records the nodes recomputed (``morse_nodes``), those that now
have a label (``labeled``), the pair and index map, the checks, the
seconds, the script, the command line, and the code commit.  The figures
are then redrawn with ``demo/replot_paper_grid.py`` (the variants the run
has), which rewrites the combined and panel figures, the attractor
lattice, and the ``figures``, ``figure_variants``, ``attractor_lattice``,
and ``figures_replotted`` records.  A summary in which
every node has a label is left unchanged.

``--workers`` evaluates the base samples in worker processes, as in the
runner; the index is computed in this process, one node at a time.
``--index-max-pieces`` bounds the pairs attempted, as in the runner (no
limit by default).

Run from the repository root, for example::

    .venv/bin/python demo/fill_missing_labels.py --workers 12 \\
        figures/paper_grid/paper-grid-impact-vdp-duffing-beta076-*-gap-refined.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from demo.replot_paper_grid import (  # noqa: E402
    ReplotError,
    _git_commit,
    problem_options,
    rebuild_problem_and_grid,
    recorded_morse_sets,
    replot,
)
from hybrid_dynamics.examples.paper_grid_examples import (  # noqa: E402
    paper_grid_problem_factory,
)
from hybrid_dynamics.src.suspension_grid_conley import (  # noqa: E402
    _rows_for_index,
    compute_suspension_grid_conley_index,
    index_pair_atoms,
    suspension_grid_gluing,
)
from hybrid_dynamics.src.suspension_grid_plot import FIGURE_VARIANTS  # noqa: E402
from hybrid_dynamics.src.suspension_grid_relation import (  # noqa: E402
    SuspensionGridRelation,
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
)

SCRIPT = "demo/fill_missing_labels.py"

#: The pair and the index map of the filled records.
FILL_INDEX_PAIR = "image"
FILL_INDEX_MAP = "auto"


class FillError(ValueError):
    """The run cannot be filled: its recomputation differs from the recorded one."""


def relation_options(summary: dict[str, object]) -> dict[str, object]:
    """Keyword arguments of :func:`compute_suspension_grid_relation` recorded in the run."""

    rule = summary["image_rule"]
    options: dict[str, object] = {
        "eval_mode": rule["eval_mode"],
        "gap_refinement_depth": int(rule.get("gap_refinement_depth") or 0),
        "exit_policy": rule.get("exit_policy", "endpoint"),
        "padding": rule.get("padding", "") != "none",
        # Summaries without an integrator record predate the batched integrator.
        "batched_flow": not str(summary.get("integrator", "SuspensionFlow")).startswith(
            "SuspensionFlow"
        ),
    }
    if rule["eval_mode"] == "random":
        options.update(
            num_pts=int(rule["num_pts"]),
            sample_depth=int(rule["sample_depth"]),
            seed=rule["seed"],
        )
    if rule["eval_mode"] == "tensor":
        options["samples_per_axis"] = int(rule["samples_per_axis"])
    return options


def _missing_nodes(summary: dict[str, object], summary_path: Path) -> list[int]:
    conley = summary.get("conley") or []
    nodes = [int(record["morse_node"]) for record in conley]
    if nodes != list(range(len(conley))):
        raise FillError(f"the index records of {summary_path} are not in node order")
    return [int(record["morse_node"]) for record in conley if not record["computed"]]


def _check_pair(record: dict[str, object], new: dict[str, object], node: int) -> None:
    """The recomputed pair of ``M(node)`` must be the recorded one."""

    for key in ("pair_atoms", "pair_pieces"):
        if key in record and record[key] != new[key]:
            raise FillError(
                f"the recomputed pair of M({node}) has {key} {new[key]!r}, the record "
                f"{record[key]!r}"
            )
    if (
        record.get("homology_computed")
        and new["homology_computed"]
        and list(record["homology_dimensions"]) != list(new["homology_dimensions"])
    ):
        raise FillError(
            f"the recomputed pair of M({node}) has homology dimensions "
            f"{new['homology_dimensions']!r}, the record {record['homology_dimensions']!r}"
        )


def _recompute_records(
    summary: dict[str, object],
    summary_path: Path,
    missing: list[int],
    *,
    workers: int,
    max_pieces: int | None,
    report,
) -> tuple[dict[int, dict[str, object]], dict[str, object], dict[str, float]]:
    """Check the recomputed relation and Morse graph; return the new records of ``missing``.

    Also returns the checks made and the seconds of each stage.  The grid,
    the relation, and the nerves are released when this returns.
    """

    seconds: dict[str, float] = {}
    started = time.perf_counter()
    problem, grid = rebuild_problem_and_grid(summary, summary_path)
    morse_sets = recorded_morse_sets(summary, summary_path, grid)
    if len(morse_sets) != len(summary["conley"]):
        raise FillError(
            f"{summary_path} has {len(summary['conley'])} index records for "
            f"{len(morse_sets)} Morse nodes"
        )
    seconds["grid"] = time.perf_counter() - started

    started = time.perf_counter()
    relation = compute_suspension_grid_relation(
        grid,
        problem,
        **relation_options(summary),
        workers=workers,
        problem_factory=paper_grid_problem_factory(summary["example"], **problem_options(summary)),
        progress=report,
    )
    seconds["relation"] = time.perf_counter() - started
    recorded_rule = summary["image_rule"].get("description")
    if relation.statistics["image_rule"] != recorded_rule:
        raise FillError(
            f"the image rule rebuilt for {summary_path} is "
            f"{relation.statistics['image_rule']!r}, the record {recorded_rule!r}"
        )
    if relation.n_edges != int(summary["relation"]["edges"]):
        raise FillError(
            f"the recomputed relation of {summary_path} has {relation.n_edges} edges, the "
            f"record {summary['relation']['edges']}"
        )
    started = time.perf_counter()
    morse = compute_suspension_morse_graph(relation)
    seconds["morse_graph"] = time.perf_counter() - started
    same_sets = len(morse.morse_sets) == len(morse_sets) and all(
        np.array_equal(new, old) for new, old in zip(morse.morse_sets, morse_sets)
    )
    if not same_sets:
        raise FillError(
            f"the recomputed Morse sets of {summary_path} differ from the recorded ones "
            f"(sizes {[int(values.size) for values in morse.morse_sets]}, recorded "
            f"{[int(values.size) for values in morse_sets]})"
        )
    edges = [list(edge) for edge in morse.edges]
    if edges != [list(edge) for edge in summary["morse_graph"]["edges"]]:
        raise FillError(
            f"the recomputed Morse graph of {summary_path} has the edges {edges}, the record "
            f"{summary['morse_graph']['edges']}"
        )
    checked = {
        "image_rule": recorded_rule,
        "relation_edges": relation.n_edges,
        "morse_sets": len(morse_sets),
        "morse_graph_edges": len(edges),
    }
    report(
        f"{summary_path}: the recomputed relation ({relation.n_edges} edges), Morse sets, "
        "and Morse graph equal the recorded ones"
    )

    # The index of the image pair reads only the rows of X = S cup F(S)
    # (the excision pair Xbar = X cup F(X) needs the images of the atoms of X).
    x_atoms = np.unique(
        np.concatenate(
            [index_pair_atoms(relation, morse_sets[node], FILL_INDEX_PAIR)[1] for node in missing]
        )
    )
    rows = _rows_for_index(relation, x_atoms)
    relation = SuspensionGridRelation(
        grid=grid, tau=relation.tau, matrix=rows, sampled=rows, statistics={}
    )
    del morse

    started = time.perf_counter()
    gluing = suspension_grid_gluing(grid)
    records: dict[int, dict[str, object]] = {}
    for node in missing:
        result = compute_suspension_grid_conley_index(
            relation,
            morse_sets[node],
            morse_node=node,
            max_pieces=max_pieces,
            gluing=gluing,
            index_pair=FILL_INDEX_PAIR,
            index_map=FILL_INDEX_MAP,
        )
        record = result.to_dict()
        _check_pair(summary["conley"][node], record, node)
        records[node] = record
        report(f"M({node}): {json.dumps(record, sort_keys=True)}")
    seconds["index"] = time.perf_counter() - started
    return records, checked, seconds


def fill_missing_labels(
    summary_path: Path,
    *,
    workers: int = 1,
    max_pieces: int | None = None,
    progress=None,
) -> dict[str, object]:
    """Compute the index of every node of one run without a label; return the entry.

    The entry is the one appended to ``labels_filled``; with no node to
    fill, the summary is unchanged and the entry has empty ``morse_nodes``.
    Raises :class:`FillError` or ``ReplotError`` (nothing changed) when the
    run cannot be rebuilt as recorded.
    """

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    missing = _missing_nodes(summary, summary_path)
    if not missing:
        return {"morse_nodes": [], "labeled": []}
    run_pair = (summary.get("conley_options") or {}).get("index_pair") or FILL_INDEX_PAIR
    if run_pair != FILL_INDEX_PAIR:
        raise FillError(
            f"the index records of {summary_path} use the {run_pair} pair; the filled records "
            f"would use the {FILL_INDEX_PAIR} pair"
        )
    records, checked, seconds = _recompute_records(
        summary, summary_path, missing, workers=workers, max_pieces=max_pieces, report=report
    )

    for node, record in records.items():
        summary["conley"][node] = record
    labeled = [node for node, record in records.items() if record["computed"]]
    if "conley_labels_in_figure" in summary:
        labels = summary["conley_labels_in_figure"]
        for node in labeled:
            labels[str(node)] = list(records[node]["shift_class"])
    entry = {
        "script": SCRIPT,
        "command_line": sys.argv[1:],
        "code": _git_commit(),
        "morse_nodes": missing,
        "labeled": labeled,
        "index_pair": FILL_INDEX_PAIR,
        "index_map": FILL_INDEX_MAP,
        "max_pieces": max_pieces,
        "checked": checked,
        "seconds": seconds,
    }
    summary.setdefault("labels_filled", []).append(entry)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replot(
        summary_path,
        variants=tuple(summary.get("figure_variants") or FIGURE_VARIANTS),
        script=SCRIPT,
    )
    return entry


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("summaries", nargs="+", type=Path, help="JSON summaries of runs")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes for the base samples of the relation (default 1)",
    )
    parser.add_argument(
        "--index-max-pieces",
        type=int,
        default=None,
        metavar="N",
        help=(
            "do not attempt the index of a Morse set whose nerve has more than N "
            "elementary pieces (default: no limit)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    failures = 0
    for path in arguments.summaries:
        try:
            entry = fill_missing_labels(
                path,
                workers=arguments.workers,
                max_pieces=arguments.index_max_pieces,
                progress=lambda message: print(message, flush=True),
            )
        except (FillError, ReplotError) as error:
            failures += 1
            print(f"refused: {error}", file=sys.stderr, flush=True)
            continue
        if not entry["morse_nodes"]:
            print(f"{path}: every Morse node has a label; unchanged", flush=True)
            continue
        print(
            f"{path}: recomputed {entry['morse_nodes']}, labeled {entry['labeled']}; "
            f"seconds {entry['seconds']}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
