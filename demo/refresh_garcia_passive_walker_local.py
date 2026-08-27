#!/usr/bin/env python3
"""Refresh derived seam, connectivity, gate, and locator audits without ODE work."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from hybrid_dynamics.examples.garcia_passive_walker_atlas import (
    AtlasWalkerCell,
    GarciaWalkerQuotientIncidence,
    garcia_passive_walker_atlas_charts,
)
from hybrid_dynamics.examples.garcia_passive_walker_local import (
    GarciaDyadicCell,
    GarciaSourceExitProvenance,
    _dyadic_rectangle_cover,
    _piece_has_unmatched_ambient_exit,
    _safe_refinement_locator,
    audit_garcia_sparse_relation_connectivity,
    attach_garcia_resume_fingerprint,
    read_garcia_local_relation,
)
from hybrid_dynamics.examples.garcia_passive_walker_suspension import (
    GarciaPassiveWalker,
)
from hybrid_dynamics.src.cmgdb_suspension_boxmap import (
    CMGDBSuspensionBoxMap,
    build_cmgdb_atlas_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("relation", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = read_garcia_local_relation(
        args.relation,
        require_resume_fingerprint=False,
    )
    metadata = payload["metadata"]
    metadata.setdefault("max_step", 0.02)
    metadata.setdefault("max_jumps", 20)
    metadata.setdefault("require_domain_path", True)
    walker = GarciaPassiveWalker(
        gamma=float(metadata["gamma"]),
        guard_delta=float(metadata["guard_delta"]),
        transversality_eta=float(metadata["transversality_eta"]),
    )
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=payload["charts"]["base"]["bounds"],
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )
    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=walker.transversality_eta,
    )
    tagged_cells = tuple(
        (
            int(raw["chart_id"]),
            int(raw["axis_depth"]),
            tuple(int(value) for value in raw["coordinates"]),
        )
        for raw in payload["cells"]
    )
    atlas = build_cmgdb_atlas_model(
        CMGDBSuspensionBoxMap(walker.system, charts, float(metadata["t_star"])),
        depth=0,
        active_dyadic_cells=tagged_cells,
    ).phaseSpace()
    cells = tuple(
        AtlasWalkerCell(
            int(raw["index"]),
            int(raw["chart_id"]),
            tuple(float(value) for value in raw["bounds"]),
        )
        for raw in payload["cells"]
    )
    relation = {
        int(raw["index"]): frozenset(int(value) for value in raw["image"])
        for raw in payload["cells"]
    }
    quotient_neighbors_by_handle: dict[int, set[int]] = {}
    for first_raw, second_raw in payload.get("quotient_neighbor_pairs", ()):
        first, second = int(first_raw), int(second_raw)
        handle = first if cells[first].chart_id == charts.handle_chart_id else second
        base = second if handle == first else first
        quotient_neighbors_by_handle.setdefault(handle, set()).add(base)
    frozen_quotient_neighbors = {
        handle: frozenset(base_cells)
        for handle, base_cells in quotient_neighbors_by_handle.items()
    }

    attachment_cache: dict[tuple[int, int], frozenset[int]] = {}
    changed_ambient_counts = 0
    provenance: dict[int, GarciaSourceExitProvenance] = {}
    for raw in payload["cells"]:
        source = int(raw["index"])
        record = raw["open_exit"]
        pieces = tuple(
            (int(item["chart_id"]), tuple(float(value) for value in item["bounds"]))
            for item in record.get("target_pieces", ())
        )
        ambient_boundary_pieces = 0
        for chart_id, bounds in pieces:
            _fine, crossings = _dyadic_rectangle_cover(
                charts,
                chart_id,
                bounds,
                int(metadata["max_axis_depth"]),
            )
            ambient_boundary_pieces += int(
                _piece_has_unmatched_ambient_exit(
                    charts,
                    incidence,
                    atlas,
                    chart_id,
                    bounds,
                    crossings,
                    pieces,
                    attachment_cache,
                    frozen_quotient_neighbors,
                )
            )
        changed_ambient_counts += int(
            ambient_boundary_pieces != int(record["ambient_boundary_pieces"])
        )
        record["ambient_boundary_pieces"] = ambient_boundary_pieces
        record.setdefault("unresolved_stage_edges", [])
        record["has_open_exit"] = bool(
            record["failed_samples_per_invocation"]
            or record["unresolved_stage_edges"]
            or record["explicit_empty_callback"]
            or record["mapgraph_empty_image"]
            or record["missing_in_domain_target_cells"]
            or ambient_boundary_pieces
            or record["wholly_outside_active_family_pieces"]
        )
        provenance[source] = GarciaSourceExitProvenance(
            source_index=source,
            callback_invocations=int(record["callback_invocations"]),
            successful_samples=int(record["successful_samples_per_invocation"]),
            failed_samples=int(record["failed_samples_per_invocation"]),
            failure_reasons=tuple(
                sorted((str(key), int(value)) for key, value in record["failure_reasons"].items())
            ),
            unresolved_stage_edges=tuple(
                tuple(int(value) for value in edge)
                for edge in record["unresolved_stage_edges"]
            ),
            returned_pieces=int(record["returned_pieces"]),
            explicit_empty_callback=bool(record["explicit_empty_callback"]),
            mapgraph_empty_image=bool(record["mapgraph_empty_image"]),
            active_target_cells=int(record["active_target_cells"]),
            missing_in_domain_target_cells=int(
                record["missing_in_domain_target_cells"]
            ),
            missing_in_domain_witnesses=tuple(
                GarciaDyadicCell(
                    int(item["chart_id"]),
                    int(item["axis_depth"]),
                    tuple(int(value) for value in item["coordinates"]),
                )
                for item in record["missing_in_domain_witnesses"]
            ),
            target_pieces=pieces,
            ambient_boundary_pieces=ambient_boundary_pieces,
            wholly_outside_active_family_pieces=int(
                record["wholly_outside_active_family_pieces"]
            ),
        )

    candidate = payload["candidate"]
    x_cells = frozenset(int(value) for value in candidate["X"])
    connectivity = audit_garcia_sparse_relation_connectivity(
        relation,
        tuple(
            GarciaDyadicCell(
                int(raw["chart_id"]),
                int(raw["axis_depth"]),
                tuple(int(value) for value in raw["coordinates"]),
            )
            for raw in payload["cells"]
        ),
        tuple(
            (int(pair[0]), int(pair[1]))
            for pair in payload.get("quotient_neighbor_pairs", ())
        ),
    )
    components = dict(connectivity.disconnected_image_components)
    disconnected = frozenset(components)
    s_cells = frozenset(int(value) for value in candidate["S"])
    a_cells = frozenset(int(value) for value in candidate["A"])
    candidate["disconnected_sources_in_S"] = sorted(s_cells & disconnected)
    candidate["disconnected_sources_in_A"] = sorted(a_cells & disconnected)
    candidate["disconnected_image_components"] = [
        {
            "source": source,
            "components": [list(component) for component in components[source]],
        }
        for source in sorted(x_cells & disconnected)
    ]
    candidate["open_exit_sources_in_S"] = [
        source for source in sorted(s_cells) if provenance[source].has_open_exit
    ]
    candidate["unresolved_stage_sources_in_S"] = [
        source
        for source in sorted(s_cells)
        if provenance[source].unresolved_stage_edges
    ]
    candidate["missing_in_domain_sources_in_S"] = [
        source
        for source in sorted(s_cells)
        if provenance[source].missing_in_domain_target_cells
    ]
    candidate["discovery_gates_passed"] = bool(
        candidate["reference_recovered"]
        and s_cells
        and not candidate["failed_sources_in_S"]
        and not candidate["unresolved_stage_sources_in_S"]
        and not candidate["open_exit_sources_in_S"]
        and not candidate["empty_sources_in_S"]
        and not candidate["disconnected_sources_in_S"]
        and not candidate["missing_in_domain_sources_in_S"]
        and not candidate["pair_second_condition_violations"]
        and not candidate["touches_nonglued_ambient_boundary"]
    )

    locator = _safe_refinement_locator(
        relation,
        provenance,
        disconnected,
        candidate["reference_source_cells"],
    )
    payload["refinement_locator"] = locator.to_dict()
    metadata["ambient_exit_semantics"] = (
        "handle s=0/1 padding is exempt only when every active seam-cell "
        "attachment is covered by returned base representatives"
    )
    metadata["derived_audits_refreshed_without_ODE_recomputation"] = True
    attach_garcia_resume_fingerprint(payload)

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.relation.suffix == ".gz":
        with gzip.open(args.relation, "wt", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        args.relation.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "relation": str(args.relation),
                "changed_source_ambient_counts": changed_ambient_counts,
                "disconnected_sources": len(disconnected),
                "safe_eligible_cells": len(locator.eligible_cells),
                "safe_recurrent_component_sizes": [
                    len(component) for component in locator.recurrent_components
                ],
                "selected_safe_component": len(locator.selected_component),
                "selected_reference_hits": len(locator.selected_reference_hits),
                "locator_seed_cells": len(locator.seed_cells),
                "open_exit_sources_in_S": len(candidate["open_exit_sources_in_S"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
