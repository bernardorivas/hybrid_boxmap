#!/usr/bin/env python3
"""Compute finite Atlas-relation indices and report continuous-system gates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from hybrid_dynamics import atlas_cells_from_phase_space
from hybrid_dynamics.examples.bouncing_ball_atlas import (
    bouncing_ball_atlas_reset_gluing,
)
from hybrid_dynamics.examples.physical_conley import (
    AtlasRelationSnapshot,
    AtlasTopCellRecord,
    audit_atlas_nerve_finite_relation,
)
from hybrid_dynamics.examples.rimless_wheel_atlas import (
    build_rimless_wheel_atlas_model,
    rimless_wheel_atlas_reset_gluing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("ball", "wheel", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/physical_conley"),
    )
    parser.add_argument(
        "--ball-relation-cache",
        type=Path,
        default=Path("data/physical_conley/ball_tau150_depth10_relation.json.gz"),
    )
    parser.add_argument(
        "--wheel-plot-cache",
        type=Path,
        default=Path("data/rimless_wheel_atlas/morse_plot_tau200_depth12.json"),
    )
    parser.add_argument(
        "--require-finite-relation-index",
        action="store_true",
        help="exit nonzero if any finite-relation index remains blocked",
    )
    return parser.parse_args()


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _run_ball(relation_cache: Path):
    _progress(f"loading bouncing-ball relation from {relation_cache}")
    started = time.perf_counter()
    snapshot = AtlasRelationSnapshot.read_gzip_json(relation_cache)
    candidate = audit_atlas_nerve_finite_relation(
        snapshot,
        morse_node=0,
        candidate_name="bouncing-ball-M0",
        gluing=bouncing_ball_atlas_reset_gluing(restitution=0.8),
    )
    summary = candidate.summary()
    summary["relation_cache"] = str(relation_cache)
    summary["elapsed_seconds"] = time.perf_counter() - started
    return candidate, summary


def _cell_key(chart_id: int, bounds) -> tuple[object, ...]:
    return (int(chart_id), *(round(float(value), 12) for value in bounds))


def _targeted_wheel_snapshot(plot_cache: Path) -> AtlasRelationSnapshot:
    """Evaluate exactly the two wheel candidate pairs, not the whole Atlas."""

    plot = json.loads(plot_cache.read_text(encoding="utf-8"))
    metadata = plot["metadata"]
    depth = int(metadata["depth"])
    t_star = float(metadata["t_star"])
    setup = build_rimless_wheel_atlas_model(
        depth=depth,
        t_star=t_star,
        samples_per_axis=3,
        padding_cells=1.0,
    )
    atlas = setup.model.phaseSpace()
    for _ in range(depth):
        atlas.subdivide()
    atlas_cells = atlas_cells_from_phase_space(atlas)
    index_by_geometry = {
        _cell_key(cell.chart_id, cell.bounds): cell.index for cell in atlas_cells
    }
    node_sets = {
        int(node["id"]): frozenset(
            index_by_geometry[_cell_key(box["chart_id"], box["bounds"])]
            for box in node["boxes"]
        )
        for node in plot["nodes"]
    }
    if set(node_sets) != {0, 1}:
        raise ValueError("wheel physical audit expects accepted nodes M(0), M(1)")

    adjacency: dict[int, tuple[int, ...]] = {}

    def evaluate(source: int) -> None:
        if source in adjacency:
            return
        cell = atlas_cells[source]
        targets = set()
        for chart_id, bounds in setup.box_map(cell.chart_id, cell.bounds):
            targets.update(
                int(target)
                for target in atlas.cover(int(chart_id), tuple(bounds))
            )
        adjacency[source] = tuple(sorted(targets))

    for source in sorted(node_sets[0] | node_sets[1]):
        evaluate(source)
    candidate_x = {
        node: node_sets[node]
        | {
            target
            for source in node_sets[node]
            for target in adjacency[source]
        }
        for node in node_sets
    }
    for source in sorted(candidate_x[0] | candidate_x[1]):
        evaluate(source)

    diagnostics = setup.box_map.diagnostics()
    failures_by_source: dict[tuple[object, ...], Counter[str]] = defaultdict(
        Counter
    )
    for diagnostic in diagnostics.retained_source_records:
        key = _cell_key(diagnostic.source_chart_id, diagnostic.source_bounds)
        failures_by_source[key].update(failure.reason for failure in diagnostic.failures)
    node_by_source = {
        source: node for node, sources in node_sets.items() for source in sources
    }
    records = []
    for cell in atlas_cells:
        failures = failures_by_source.get(
            _cell_key(cell.chart_id, cell.bounds), Counter()
        )
        records.append(
            AtlasTopCellRecord(
                index=cell.index,
                chart_id=cell.chart_id,
                bounds=cell.bounds,
                image=adjacency.get(cell.index, ()),
                morse_node=node_by_source.get(cell.index),
                relation_evaluated=cell.index in adjacency,
                callback_failed_samples=sum(failures.values()),
                callback_failure_reasons=tuple(sorted(failures.items())),
            )
        )
    return AtlasRelationSnapshot(
        model="rimless-wheel",
        depth=depth,
        t_star=t_star,
        cells=tuple(records),
        morse_nodes=tuple(sorted(node_sets)),
        morse_edges=tuple(
            (int(source), int(target)) for source, target in plot["edges"]
        ),
        base_chart_id=0,
        handle_chart_id=1,
        whole_cell_outer_enclosure_certified=False,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=int(diagnostics.failed_samples),
        box_map_empty_images=int(diagnostics.empty_images),
        box_map_unresolved_stage_edges=int(diagnostics.unresolved_stage_edges),
        relation_scope="targeted_index_candidates",
    )


def _run_wheel(output_dir: Path, plot_cache: Path):
    _progress("evaluating wheel M(0) and M(1) candidate relations only")
    started = time.perf_counter()
    snapshot = _targeted_wheel_snapshot(plot_cache)
    cache_path = output_dir / "wheel_tau200_depth12_targeted_relation.json.gz"
    snapshot.write_gzip_json(cache_path)
    _progress(f"wrote targeted wheel relation and exits to {cache_path}")
    gluing = rimless_wheel_atlas_reset_gluing(alpha=0.4, gamma=0.2)
    gait = audit_atlas_nerve_finite_relation(
        snapshot,
        morse_node=0,
        candidate_name="rimless-wheel-gait-M0",
        gluing=gluing,
    )
    saddle = audit_atlas_nerve_finite_relation(
        snapshot,
        morse_node=1,
        candidate_name="rimless-wheel-saddle-M1",
        gluing=gluing,
    )
    elapsed = time.perf_counter() - started
    summaries = []
    for candidate in (gait, saddle):
        summary = candidate.summary()
        summary["relation_cache"] = str(cache_path)
        summary["shared_model_elapsed_seconds"] = elapsed
        summaries.append(summary)
    return (gait, saddle), summaries


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    summaries = []
    if args.model in {"ball", "all"}:
        candidate, summary = _run_ball(args.ball_relation_cache)
        candidates.append(candidate)
        summaries.append(summary)
    if args.model in {"wheel", "all"}:
        wheel_candidates, wheel_summaries = _run_wheel(
            args.output_dir, args.wheel_plot_cache
        )
        candidates.extend(wheel_candidates)
        summaries.extend(wheel_summaries)

    report = {
        "schema": "physical-conley-finite-relation-audit-v2",
        "computed_at": date.today().isoformat(),
        "method": {
            "top_pair": "CMGDB X=S union F(S), A=F(S) minus S",
            "cellular_pair": "induced reset-quotient nerve pair P1=N(X), P0=N(A)",
            "carrier": "face-nested acyclic carrier from actual MapGraph over GF(5)",
            "finite_relation_policy": "compute only after finite pair/carrier/chain gates",
            "continuous_policy": "report separately; never infer from finite relation",
            "analytic_labels_used": False,
        },
        "candidates": summaries,
        "computed_finite_relation_indices": sum(
            candidate.finite_relation_conley_index_computed for candidate in candidates
        ),
        "withheld_finite_relation_indices": sum(
            not candidate.finite_relation_conley_index_computed for candidate in candidates
        ),
        "certified_continuous_system_indices": 0,
    }
    report_path = args.output_dir / f"physical_conley_audit_{args.model}.json"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    _progress(f"wrote physical Conley audit to {report_path}")

    if args.require_finite_relation_index and any(
        not candidate.finite_relation_conley_index_computed for candidate in candidates
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
