#!/usr/bin/env python3
"""Redraw the figures of paper-grid runs from their JSON summaries.

For each JSON summary written by ``demo/run_paper_grid_examples.py`` this
script rebuilds the grid ``Xi_n`` of the example (or variant) and level (a
deterministic construction; the rebuilt grid must match the ``grid`` record of
the run), reads the Morse sets (``morse_graph.morse_set_atoms``), the Morse
graph, and the index records (``conley``), and writes the figure variants next
to the JSON (or to ``--output-dir``): ``all`` (``<stem>.pdf``/``.png``) and
``nontrivial`` (``<stem>-nontrivial.pdf``/``.png``), and each panel of a
variant as its own figure (``<variant stem>-base``, ``-zoom-A``, ...,
``-handle``, ``-graph``, PDF and PNG).  No dynamics is resampled.  The
``figures`` and ``figure_variants`` records of the JSON are updated to the
files written (the panel files under ``panel_files`` of each variant; older
summaries have none), and ``figures_replotted`` records the code.

Summaries written before the runner stored the Morse sets have no
``morse_graph.morse_set_atoms`` and cannot be redrawn; rerun them.

Run from the ``code`` directory, for example::

    .venv/bin/python demo/replot_paper_grid.py figures/paper_grid/*-corners.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from hybrid_dynamics.examples.paper_grid_examples import paper_grid_problem  # noqa: E402
from hybrid_dynamics.examples.paper_grid_figures import (  # noqa: E402
    write_paper_grid_figures,
)
from hybrid_dynamics.src.suspension_grid import build_suspension_grid  # noqa: E402
from hybrid_dynamics.src.suspension_grid_plot import (  # noqa: E402
    FIGURE_VARIANTS,
    decode_index_ranges,
)


class ReplotError(ValueError):
    """The JSON summary does not contain what the figures need."""


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(CODE_ROOT))
    except ValueError:
        return str(path)


def _git_commit() -> dict[str, object]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=CODE_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    # The redrawn figures and summaries are tracked outputs under figures/;
    # only changes elsewhere make the code differ from the commit.
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(
            run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)figures")
        ),
    }


def replot(
    summary_path: Path,
    *,
    variants: tuple[str, ...] = FIGURE_VARIANTS,
    output_dir: Path | None = None,
    update_json: bool = True,
) -> dict[str, object]:
    """Redraw the figure variants of one run; return the figure records."""

    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    morse_graph = summary.get("morse_graph", {})
    encoded = morse_graph.get("morse_set_atoms")
    if encoded is None:
        raise ReplotError(
            f"{summary_path} has no morse_graph.morse_set_atoms (written before the runner "
            "stored the Morse sets); rerun it with demo/run_paper_grid_examples.py"
        )
    example = summary["example"]
    level = int(summary["level"])
    tau = float(summary["tau"])
    level_offset = int(summary.get("level_offset", 0))
    options: dict[str, object] = {"tau": tau}
    if level_offset:
        options["level_offset"] = level_offset
    problem = paper_grid_problem(example, **options)
    grid = build_suspension_grid(problem.window, problem.guard, level)
    rebuilt = json.loads(json.dumps(grid.summary()))
    if rebuilt != summary["grid"]:
        differing = sorted(
            key
            for key in set(rebuilt) | set(summary["grid"])
            if rebuilt.get(key) != summary["grid"].get(key)
        )
        raise ReplotError(
            f"the rebuilt grid of {summary_path} differs from the recorded one in {differing!r}"
        )
    morse_sets = [decode_index_ranges(text) for text in encoded]
    recorded_sizes = [int(node["atoms"]) for node in morse_graph["nodes"]]
    if [values.size for values in morse_sets] != recorded_sizes:
        raise ReplotError(f"the Morse sets of {summary_path} do not match their recorded sizes")
    if morse_sets and max(int(values.max()) for values in morse_sets if values.size) >= grid.n_atoms:
        raise ReplotError(f"the Morse sets of {summary_path} refer to atoms outside the grid")
    edges = [tuple(edge) for edge in morse_graph["edges"]]
    conley = summary.get("conley", [])
    if "nontrivial" in variants and len(conley) != len(morse_sets):
        raise ReplotError(
            f"{summary_path} has {len(conley)} index records for {len(morse_sets)} Morse "
            "nodes; the nontrivial variant needs one per node"
        )

    directory = summary_path.parent if output_dir is None else Path(output_dir)
    figures = write_paper_grid_figures(
        grid,
        morse_sets,
        edges,
        conley,
        example=example,
        tau=tau,
        level=level,
        output_stem=directory / summary_path.stem,
        audit_path=summary_path,
        variants=variants,
        display_path=_display_path,
    )
    if update_json:
        summary["figures"] = figures["figures"]
        summary["figure_variants"] = figures["figure_variants"]
        summary["figures_replotted"] = {
            "script": "demo/replot_paper_grid.py",
            "command_line": sys.argv[1:],
            "code": _git_commit(),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return figures


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path, help="JSON summaries of runs")
    parser.add_argument(
        "--figure-variants",
        default=",".join(FIGURE_VARIANTS),
        metavar="VARIANT[,VARIANT]",
        help=f"comma-separated figure variants from {', '.join(FIGURE_VARIANTS)} (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory of the figures (default: the directory of each JSON)",
    )
    parser.add_argument(
        "--no-update-json",
        action="store_true",
        help="leave the JSON summaries unchanged",
    )
    arguments = parser.parse_args()
    variants = tuple(
        dict.fromkeys(
            value.strip() for value in arguments.figure_variants.split(",") if value.strip()
        )
    )
    if not variants or any(value not in FIGURE_VARIANTS for value in variants):
        parser.error(
            f"--figure-variants takes values from {', '.join(FIGURE_VARIANTS)}; "
            f"got {arguments.figure_variants!r}"
        )
    arguments.figure_variants = variants
    return arguments


def main() -> int:
    arguments = _arguments()
    failures = 0
    for path in arguments.summaries:
        try:
            figures = replot(
                path,
                variants=arguments.figure_variants,
                output_dir=arguments.output_dir,
                update_json=not arguments.no_update_json,
            )
        except ReplotError as error:
            failures += 1
            print(f"skipped: {error}", file=sys.stderr, flush=True)
            continue
        for variant, record in figures["figure_variants"].items():
            print(
                f"{path}: {variant}: {len(record['shown_nodes'])} shown, "
                f"{len(record['hidden_nodes'])} hidden, "
                f"{len(record['blocked_nodes_shown'])} blocked; {record['files']}; "
                f"panels {sorted(record['panel_files'])}",
                flush=True,
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
