"""Dense-reference local tube for the Garcia passive walker.

The construction in this module is geometry-only apart from one inexpensive
reference integration.  It never evaluates a source box and never constructs
a ``MapGraph``.  Base and handle tubes are formed independently around the
complete two-stride reference trajectory.  Quotient compatibility is then
handled by one nonpercolating attachment pass: every selected handle cell on
``s=0`` or ``s=1`` contributes its complete analytic guard/reset carrier in
the base chart.  Newly added base cells do not feed back into remote handle
cofaces.  Those reverse cofaces remain an explicitly open boundary.

This distinction is essential.  Recursively closing the graph of *closed
top-cell incidences* percolates through shared lower-dimensional faces and can
import nearly the entire remote guard seam.  The symbolic attachment records
below retain the lower-dimensional topology without making that mistake.
"""

from __future__ import annotations

import hashlib
import gzip
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .garcia_passive_walker_atlas import (
    AtlasWalkerCell,
    GarciaWalkerQuotientIncidence,
    garcia_passive_walker_atlas_charts,
)
from .garcia_passive_walker_local import (
    GarciaDyadicCell,
    GarciaMixedDyadicFamily,
    _dyadic_rectangle_cover,
    garcia_non_domain_failure_sources,
)
from .garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
    PERIOD_TWO_POINT_A,
    PERIOD_TWO_POINT_B,
    GarciaPassiveWalker,
    post_impact_state,
)


TUBE_SCHEMA: Final = "garcia-walker-dense-orbit-tube-v1"
ATTACHMENT_SCHEMA: Final = "garcia-symbolic-handle-attachment-carriers-v1"
DEFAULT_TUBE_SHA256: Final = (
    "8b762570020296fb80aaa4160c2a72bfd8645972c94e7e7d5e5bedb43a85f16a"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _flat_code(coordinates: tuple[int, int, int, int], subdivisions: int) -> int:
    first, second, third, fourth = coordinates
    return int(
        ((first * subdivisions + second) * subdivisions + third)
        * subdivisions
        + fourth
    )


def _coordinates(code: int, subdivisions: int) -> tuple[int, int, int, int]:
    fourth = int(code % subdivisions)
    code //= subdivisions
    third = int(code % subdivisions)
    code //= subdivisions
    second = int(code % subdivisions)
    code //= subdivisions
    return int(code), second, third, fourth


def _closed_tube_mask(
    points: np.ndarray,
    chart_bounds: tuple[tuple[float, float], ...],
    *,
    axis_depth: int,
    normalized_radius: float,
) -> np.ndarray:
    """Cover the closed normalized L-infinity tube by uniform closed cells."""

    bounds = np.asarray(chart_bounds, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.all(np.isfinite(values)):
        raise ValueError("dense tube points must be a finite N-by-4 array")
    subdivisions = 2**axis_depth
    normalized = (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    if np.any(normalized < -1.0e-9) or np.any(normalized > 1.0 + 1.0e-9):
        raise ValueError("dense reference trace leaves its declared Atlas chart")
    tolerance = 1.0e-12
    lower = np.maximum(
        0,
        np.ceil(subdivisions * (normalized - normalized_radius) - tolerance).astype(
            np.int64
        )
        - 1,
    )
    upper = np.minimum(
        subdivisions - 1,
        np.floor(
            subdivisions * (normalized + normalized_radius) + tolerance
        ).astype(np.int64),
    )
    if np.any(lower > upper):
        raise ValueError("dense tube has an empty clipped cell carrier")

    # Four-dimensional imos/difference-array update: every point contributes
    # one axis-aligned integer box using only 2^4 scatter operations.  This is
    # exact for the closed-cell rule and avoids point-by-cell loops.
    difference = np.zeros((subdivisions + 1,) * 4, dtype=np.int32)
    for corner in itertools.product((0, 1), repeat=4):
        indices = tuple(
            np.where(corner[axis], upper[:, axis] + 1, lower[:, axis])
            for axis in range(4)
        )
        np.add.at(difference, indices, -1 if sum(corner) % 2 else 1)
    for axis in range(4):
        difference = np.cumsum(difference, axis=axis)
    return difference[(slice(0, subdivisions),) * 4] > 0


def _dense_reference_points(
    trajectory: object,
    charts: object,
    *,
    base_samples_per_stride: int,
    handle_samples_per_jump: int,
) -> tuple[np.ndarray, np.ndarray]:
    segments = trajectory.segments[:2]
    jumps = trajectory.jump_states[:2]
    base = np.concatenate(
        [
            np.asarray(
                segment.solution(
                    np.linspace(
                        segment.t_start,
                        segment.t_end,
                        base_samples_per_stride,
                    )
                ),
                dtype=np.float64,
            ).T
            for segment in segments
        ]
    )
    phases = np.linspace(0.0, 1.0, handle_samples_per_jump)
    handle = np.concatenate(
        [
            np.asarray(
                [charts.encode_handle(guard, float(phase)) for phase in phases],
                dtype=np.float64,
            )
            for guard, _reset in jumps
        ]
    )
    return base, handle


def _handle_attachment_carrier(
    token: GarciaDyadicCell,
    *,
    charts: object,
    incidence: GarciaWalkerQuotientIncidence,
) -> tuple[GarciaDyadicCell, ...]:
    """Return every base cell meeting the complete selected handle face."""

    subdivisions = 2**token.axis_depth
    if token.chart_id != charts.handle_chart_id or token.coordinates[-1] not in (
        0,
        subdivisions - 1,
    ):
        return ()
    handle = AtlasWalkerCell(-1, token.chart_id, token.bounds(charts))
    hull = incidence.attachment_hull(
        handle,
        -1 if token.coordinates[-1] == 0 else 1,
    )
    candidates, crossings = _dyadic_rectangle_cover(
        charts,
        charts.base_chart_id,
        hull,
        token.axis_depth,
    )
    if any(crossings):
        raise ValueError("complete handle attachment carrier leaves the base chart")
    result = tuple(
        sorted(
            candidate
            for candidate in candidates
            if incidence.intersects(
                AtlasWalkerCell(-2, candidate.chart_id, candidate.bounds(charts)),
                handle,
            )
        )
    )
    return result


@dataclass(frozen=True)
class GarciaOrbitTubeConstruction:
    family: GarciaMixedDyadicFamily
    normalized_radius: float
    axis_depth: int
    reference_provenance: dict[str, object]
    attachment_carriers: tuple[
        tuple[GarciaDyadicCell, tuple[GarciaDyadicCell, ...]], ...
    ]
    raw_base_cells: int
    raw_handle_cells: int
    added_base_attachment_cells: int
    active_quotient_incidences: int
    reverse_open_quotient_incidences: int
    same_chart_open_boundary_cells: tuple[GarciaDyadicCell, ...]
    reverse_open_base_boundary_cells: tuple[GarciaDyadicCell, ...]

    def to_dict(self) -> dict[str, object]:
        cells = [
            {
                "chart_id": cell.chart_id,
                "axis_depth": cell.axis_depth,
                "coordinates": list(cell.coordinates),
            }
            for cell in self.family.cells
        ]
        carriers = [
            {
                "handle": {
                    "chart_id": handle.chart_id,
                    "axis_depth": handle.axis_depth,
                    "coordinates": list(handle.coordinates),
                    "face": (
                        "guard_s0" if handle.coordinates[-1] == 0 else "reset_s1"
                    ),
                },
                "base": [
                    {
                        "chart_id": base.chart_id,
                        "axis_depth": base.axis_depth,
                        "coordinates": list(base.coordinates),
                    }
                    for base in bases
                ],
            }
            for handle, bases in self.attachment_carriers
        ]
        family_hash = hashlib.sha256(_canonical_json(cells)).hexdigest()
        carrier_hash = hashlib.sha256(_canonical_json(carriers)).hexdigest()
        payload: dict[str, object] = {
            "schema": TUBE_SCHEMA,
            "construction_role": "local_orbit_validation_not_discovery",
            "no_box_map_or_MapGraph_dynamics_evaluated": True,
            "reference_only_simulation_evaluated": True,
            "scientific_result_accepted": False,
            "whole_cell_outer_enclosure_certified": False,
            "axis_depth": self.axis_depth,
            "total_depth": 4 * self.axis_depth,
            "normalized_Linf_radius": self.normalized_radius,
            "attachment_policy": (
                "selected_handle_s0_s1_to_complete_base_carrier_one_pass_"
                "without_base_to_handle_feedback"
            ),
            "attachment_padding_cells": 0,
            "reverse_extra_seam_cofaces_are_open": True,
            "raw_tube_cells": {
                "base": self.raw_base_cells,
                "handle": self.raw_handle_cells,
            },
            "added_base_attachment_cells": self.added_base_attachment_cells,
            "active_cells": len(self.family.cells),
            "active_chart_counts": self.family.chart_counts(),
            "active_quotient_incidences": self.active_quotient_incidences,
            "reverse_open_quotient_incidences": (
                self.reverse_open_quotient_incidences
            ),
            "same_chart_open_boundary_cells": [
                {
                    "chart_id": cell.chart_id,
                    "axis_depth": cell.axis_depth,
                    "coordinates": list(cell.coordinates),
                }
                for cell in self.same_chart_open_boundary_cells
            ],
            "reverse_open_base_boundary_cells": [
                {
                    "chart_id": cell.chart_id,
                    "axis_depth": cell.axis_depth,
                    "coordinates": list(cell.coordinates),
                }
                for cell in self.reverse_open_base_boundary_cells
            ],
            "selected_handle_seam_cells": len(self.attachment_carriers),
            "selected_handle_seam_faces_without_full_base_carrier": sum(
                not bases for _handle, bases in self.attachment_carriers
            ),
            "family_cells_sha256": family_hash,
            "attachment_carriers": {
                "schema": ATTACHMENT_SCHEMA,
                "records": carriers,
                "sha256": carrier_hash,
            },
            "reference_trace": dict(self.reference_provenance),
            "cells": cells,
        }
        fingerprint_fields = dict(payload)
        payload["fingerprint"] = {
            "algorithm": "sha256",
            "sha256": hashlib.sha256(
                _canonical_json(fingerprint_fields)
            ).hexdigest(),
        }
        return payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        if target.suffix == ".gz":
            with gzip.open(target, "wt", encoding="utf-8") as stream:
                stream.write(encoded)
        else:
            target.write_text(encoded, encoding="utf-8")
        return target


@dataclass(frozen=True)
class GarciaOrbitTubeBoundaryAudit:
    same_chart_boundary_cells: frozenset[int]
    reverse_seam_boundary_cells: frozenset[int]
    same_chart_witnesses_in_s: frozenset[int]
    reverse_seam_witnesses_in_s: frozenset[int]
    ambient_witnesses_in_s: frozenset[int]

    @property
    def passed(self) -> bool:
        return not (
            self.same_chart_witnesses_in_s
            or self.reverse_seam_witnesses_in_s
            or self.ambient_witnesses_in_s
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": (
                "S must avoid the closed same-chart one-ring boundary and all "
                "active base carriers incident to omitted reverse seam cofaces; "
                "L=A may touch or exit"
            ),
            "same_chart_closed_one_ring_boundary_cells": len(
                self.same_chart_boundary_cells
            ),
            "reverse_open_quotient_boundary_cells": len(
                self.reverse_seam_boundary_cells
            ),
            "same_chart_boundary_witnesses_in_S": sorted(
                self.same_chart_witnesses_in_s
            ),
            "reverse_seam_boundary_witnesses_in_S": sorted(
                self.reverse_seam_witnesses_in_s
            ),
            "ambient_boundary_witnesses_in_S": sorted(
                self.ambient_witnesses_in_s
            ),
            "passed": self.passed,
        }


def audit_garcia_orbit_tube_candidate_boundary(
    construction: GarciaOrbitTubeConstruction,
    dyadic_cells: tuple[GarciaDyadicCell, ...],
    s_cells: frozenset[int],
    ambient_boundary_witnesses: tuple[int, ...] = (),
) -> GarciaOrbitTubeBoundaryAudit:
    """Check that the recurrent set lies in the local tube interior."""

    if tuple(sorted(dyadic_cells)) != construction.family.cells:
        raise ValueError("candidate Atlas order does not match the pinned tube family")
    if any(index < 0 or index >= len(dyadic_cells) for index in s_cells):
        raise ValueError("candidate S references a cell outside the tube")
    index_by_cell = {cell: index for index, cell in enumerate(dyadic_cells)}
    same = frozenset(
        index_by_cell[cell] for cell in construction.same_chart_open_boundary_cells
    )
    reverse = frozenset(
        index_by_cell[cell]
        for cell in construction.reverse_open_base_boundary_cells
    )
    ambient = frozenset(int(value) for value in ambient_boundary_witnesses)
    if any(index < 0 or index >= len(dyadic_cells) for index in ambient):
        raise ValueError("ambient-boundary witness lies outside the tube")
    return GarciaOrbitTubeBoundaryAudit(
        same_chart_boundary_cells=same,
        reverse_seam_boundary_cells=reverse,
        same_chart_witnesses_in_s=s_cells & same,
        reverse_seam_witnesses_in_s=s_cells & reverse,
        ambient_witnesses_in_s=s_cells & ambient,
    )


def garcia_orbit_tube_acceptance_blockers(
    run: object,
    support_audit: object,
    *,
    same_chart_boundary_in_s: set[int],
    reverse_seam_boundary_in_s: set[int],
) -> list[str]:
    """Apply the predeclared open-pair acceptance policy to one tube run.

    Every defect in ``S=N\\L`` blocks.  In ``L=A``, explicit domain exits and
    disconnected or empty represented values may remain exits when the finite
    pair condition holds; unresolved stages and non-domain failures still
    block because their images could re-enter ``S``.
    """

    candidate = run.candidate
    result = list(support_audit.terminal_blockers)
    checks = {
        "reference_endpoint_miss": candidate.reference_endpoint_misses,
        "reference_evaluation_failure": candidate.reference_evaluation_failures,
        "failed_source_in_S": len(candidate.failed_sources_in_s),
        "unresolved_stage_source_in_S": len(
            candidate.unresolved_stage_sources_in_s
        ),
        "open_exit_source_in_S": len(candidate.open_exit_sources_in_s),
        "empty_source_in_S": len(candidate.empty_sources_in_s),
        "disconnected_source_in_S": len(candidate.disconnected_sources_in_s),
        "missing_support_source_in_S": len(
            candidate.missing_in_domain_sources_in_s
        ),
        "nonglued_boundary_source_in_S": len(
            candidate.touches_nonglued_ambient_boundary
        ),
        "pair_second_condition_violation": len(
            candidate.pair_second_condition_violations
        ),
        "support_not_saturated_in_N_minus_L": len(support_audit.added_cells),
    }
    result.extend(label for label, count in checks.items() if count)
    if same_chart_boundary_in_s:
        result.append("S_touches_same_chart_closed_one_ring_boundary")
    if reverse_seam_boundary_in_s:
        result.append("S_touches_reverse_open_quotient_boundary")
    if not candidate.reference_recovered:
        result.append("reference_gait_not_recovered")
    if any(
        run.source_provenance[source].unresolved_stage_edges
        for source in candidate.a_cells
    ):
        result.append("unresolved_stage_source_in_A")
    if garcia_non_domain_failure_sources(
        run.source_provenance, candidate.a_cells
    ):
        result.append("non_domain_sample_failure_in_A")
    return sorted(set(result))


def read_garcia_orbit_tube(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> GarciaOrbitTubeConstruction:
    """Strictly validate a persisted tube without evaluating any dynamics."""

    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema",
        "construction_role",
        "no_box_map_or_MapGraph_dynamics_evaluated",
        "reference_only_simulation_evaluated",
        "scientific_result_accepted",
        "whole_cell_outer_enclosure_certified",
        "axis_depth",
        "total_depth",
        "normalized_Linf_radius",
        "attachment_policy",
        "attachment_padding_cells",
        "reverse_extra_seam_cofaces_are_open",
        "raw_tube_cells",
        "added_base_attachment_cells",
        "active_cells",
        "active_chart_counts",
        "active_quotient_incidences",
        "reverse_open_quotient_incidences",
        "same_chart_open_boundary_cells",
        "reverse_open_base_boundary_cells",
        "selected_handle_seam_cells",
        "selected_handle_seam_faces_without_full_base_carrier",
        "family_cells_sha256",
        "attachment_carriers",
        "reference_trace",
        "cells",
        "fingerprint",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("orbit-tube artifact is not canonical")
    if (
        payload["schema"] != TUBE_SCHEMA
        or payload["construction_role"] != "local_orbit_validation_not_discovery"
        or payload["no_box_map_or_MapGraph_dynamics_evaluated"] is not True
        or payload["reference_only_simulation_evaluated"] is not True
        or payload["scientific_result_accepted"] is not False
        or payload["whole_cell_outer_enclosure_certified"] is not False
        or payload["attachment_policy"]
        != "selected_handle_s0_s1_to_complete_base_carrier_one_pass_without_base_to_handle_feedback"
        or payload["attachment_padding_cells"] != 0
        or payload["reverse_extra_seam_cofaces_are_open"] is not True
    ):
        raise ValueError("orbit-tube artifact makes incompatible scope claims")
    fingerprint = payload["fingerprint"]
    unsigned = {key: value for key, value in payload.items() if key != "fingerprint"}
    if (
        not isinstance(fingerprint, dict)
        or set(fingerprint) != {"algorithm", "sha256"}
        or fingerprint["algorithm"] != "sha256"
        or fingerprint["sha256"] != hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    ):
        raise ValueError("orbit-tube artifact fingerprint is invalid")
    if expected_fingerprint is not None and fingerprint["sha256"] != expected_fingerprint:
        raise ValueError("orbit-tube artifact differs from the pinned construction")
    axis_depth = payload["axis_depth"]
    radius = payload["normalized_Linf_radius"]
    if (
        type(axis_depth) is not int
        or axis_depth < 1
        or axis_depth > 5
        or payload["total_depth"] != 4 * axis_depth
        or type(radius) not in (int, float)
        or not np.isfinite(float(radius))
        or not 0 < float(radius) < 0.5
    ):
        raise ValueError("orbit-tube depth or radius is invalid")
    raw_cells = payload["cells"]
    if not isinstance(raw_cells, list):
        raise ValueError("orbit-tube cell table is malformed")

    def token(raw: object) -> GarciaDyadicCell:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"chart_id", "axis_depth", "coordinates"}
            or type(raw["chart_id"]) is not int
            or type(raw["axis_depth"]) is not int
            or raw["axis_depth"] != axis_depth
            or not isinstance(raw["coordinates"], list)
            or len(raw["coordinates"]) != 4
            or any(type(value) is not int for value in raw["coordinates"])
        ):
            raise ValueError("orbit-tube dyadic token is malformed")
        return GarciaDyadicCell(
            raw["chart_id"], raw["axis_depth"], tuple(raw["coordinates"])
        )

    cells = tuple(token(raw) for raw in raw_cells)
    if cells != tuple(sorted(set(cells))) or any(
        cell.axis_depth != axis_depth for cell in cells
    ):
        raise ValueError("orbit-tube cells are not a canonical uniform antichain")
    if payload["family_cells_sha256"] != hashlib.sha256(
        _canonical_json(raw_cells)
    ).hexdigest():
        raise ValueError("orbit-tube family hash is invalid")
    family = GarciaMixedDyadicFamily(
        cells,
        role=(
            "dense two-stride orbit-validation tube; normalized L-infinity radius "
            f"{float(radius):g}; one-pass handle-face attachment; open complement"
        ),
    )
    if payload["active_cells"] != len(cells) or {
        str(key): value for key, value in family.chart_counts().items()
    } != {str(key): value for key, value in payload["active_chart_counts"].items()}:
        raise ValueError("orbit-tube active-cell summary is inconsistent")
    reference = payload["reference_trace"]
    reference_keys = {
        "construction",
        "gamma",
        "guard_delta",
        "transversality_eta",
        "domain_bounds",
        "period_two_point_A",
        "period_two_point_B",
        "initial_full_state",
        "time_span",
        "max_jumps_requested",
        "completed_jumps",
        "rtol",
        "atol",
        "max_step",
        "dense_output",
        "segment_intervals",
        "guard_states",
        "reset_states",
        "sampling_convergence",
        "sampling_stable_at_final_doubling",
        "final_base_samples_per_stride",
        "final_handle_samples_per_jump",
        "dense_coordinate_hash_rule",
        "dense_coordinate_sha256",
    }
    if not isinstance(reference, dict) or set(reference) != reference_keys:
        raise ValueError("orbit-tube reference provenance is malformed")
    if (
        reference["construction"]
        != "reference-only dense integration of the stored period-two seed for two continuous strides plus both intrinsic handle traversals"
        or reference["dense_output"] is not True
        or reference["sampling_stable_at_final_doubling"] is not True
        or reference["max_jumps_requested"] != 2
        or reference["completed_jumps"] < 2
        or reference["dense_coordinate_hash_rule"]
        != "SHA-256 of base then handle C-order raw little-endian float64 bytes"
        or not isinstance(reference["dense_coordinate_sha256"], str)
        or len(reference["dense_coordinate_sha256"]) != 64
    ):
        raise ValueError("orbit-tube reference provenance has incompatible semantics")
    walker = GarciaPassiveWalker(
        gamma=float(reference["gamma"]),
        guard_delta=float(reference["guard_delta"]),
        transversality_eta=float(reference["transversality_eta"]),
        domain_bounds=[tuple(map(float, value)) for value in reference["domain_bounds"]],
        max_jumps=20,
    )
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=walker.guard_delta,
        transversality_eta=walker.transversality_eta,
    )
    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=walker.transversality_eta,
    )
    carrier_payload = payload["attachment_carriers"]
    if (
        not isinstance(carrier_payload, dict)
        or set(carrier_payload) != {"schema", "records", "sha256"}
        or carrier_payload["schema"] != ATTACHMENT_SCHEMA
        or not isinstance(carrier_payload["records"], list)
        or carrier_payload["sha256"]
        != hashlib.sha256(_canonical_json(carrier_payload["records"])).hexdigest()
    ):
        raise ValueError("orbit-tube attachment carrier table is invalid")
    active = frozenset(cells)
    carriers: list[tuple[GarciaDyadicCell, tuple[GarciaDyadicCell, ...]]] = []
    seen_handles: set[GarciaDyadicCell] = set()
    for record in carrier_payload["records"]:
        if not isinstance(record, dict) or set(record) != {"handle", "base"}:
            raise ValueError("orbit-tube attachment record is malformed")
        handle_raw = record["handle"]
        if not isinstance(handle_raw, dict) or set(handle_raw) != {
            "chart_id",
            "axis_depth",
            "coordinates",
            "face",
        }:
            raise ValueError("orbit-tube handle attachment is malformed")
        handle = token({key: handle_raw[key] for key in ("chart_id", "axis_depth", "coordinates")})
        bases = tuple(token(value) for value in record["base"])
        subdivisions = 2**axis_depth
        expected_face = "guard_s0" if handle.coordinates[-1] == 0 else "reset_s1"
        if (
            handle in seen_handles
            or handle not in active
            or handle.chart_id != charts.handle_chart_id
            or handle.coordinates[-1] not in (0, subdivisions - 1)
            or handle_raw["face"] != expected_face
            or bases != tuple(sorted(set(bases)))
            or any(base not in active or base.chart_id != charts.base_chart_id for base in bases)
            or bases
            != _handle_attachment_carrier(handle, charts=charts, incidence=incidence)
        ):
            raise ValueError("orbit-tube attachment carrier is incomplete or inconsistent")
        seen_handles.add(handle)
        carriers.append((handle, bases))
    expected_handles = {
        cell
        for cell in cells
        if cell.chart_id == charts.handle_chart_id
        and cell.coordinates[-1] in (0, 2**axis_depth - 1)
    }
    if seen_handles != expected_handles:
        raise ValueError("orbit-tube does not record every selected handle seam face")
    active_pairs = sum(len(bases) for _handle, bases in carriers)
    if (
        payload["active_quotient_incidences"] != active_pairs
        or payload["selected_handle_seam_cells"] != len(carriers)
        or payload["selected_handle_seam_faces_without_full_base_carrier"]
        != sum(not bases for _handle, bases in carriers)
    ):
        raise ValueError("orbit-tube attachment summary is inconsistent")
    chart_counts = family.chart_counts()
    raw_summary = payload["raw_tube_cells"]
    if (
        not isinstance(raw_summary, dict)
        or set(raw_summary) != {"base", "handle"}
        or type(payload["added_base_attachment_cells"]) is not int
        or raw_summary["base"] + payload["added_base_attachment_cells"]
        != chart_counts[charts.base_chart_id]
        or raw_summary["handle"] != chart_counts[charts.handle_chart_id]
    ):
        raise ValueError("orbit-tube raw/addition counts are inconsistent")
    reverse_open = 0
    reverse_open_base: set[GarciaDyadicCell] = set()
    subdivisions = 2**axis_depth
    for coordinates in itertools.product(range(subdivisions), repeat=3):
        for phase_index in (0, subdivisions - 1):
            handle = GarciaDyadicCell(
                charts.handle_chart_id,
                axis_depth,
                (*coordinates, phase_index),
            )
            if handle in active:
                continue
            for base in _handle_attachment_carrier(
                handle, charts=charts, incidence=incidence
            ):
                if base in active:
                    reverse_open += 1
                    reverse_open_base.add(base)
    if payload["reverse_open_quotient_incidences"] != reverse_open:
        raise ValueError("orbit-tube reverse open-boundary count is inconsistent")
    stored_reverse_base = tuple(token(raw) for raw in payload["reverse_open_base_boundary_cells"])
    if stored_reverse_base != tuple(sorted(reverse_open_base)):
        raise ValueError("orbit-tube reverse open-boundary carrier set is inconsistent")
    subdivisions = 2**axis_depth
    same_chart_boundary: set[GarciaDyadicCell] = set()
    glued_handles = {handle for handle, bases in carriers if bases}
    offsets = tuple(
        offset
        for offset in itertools.product((-1, 0, 1), repeat=4)
        if any(offset)
    )
    for cell in active:
        for offset in offsets:
            neighbor_coordinates = tuple(
                coordinate + delta
                for coordinate, delta in zip(cell.coordinates, offset, strict=True)
            )
            outside_axes = {
                axis
                for axis, coordinate in enumerate(neighbor_coordinates)
                if coordinate < 0 or coordinate >= subdivisions
            }
            if outside_axes:
                if cell in glued_handles and outside_axes == {3}:
                    continue
                same_chart_boundary.add(cell)
                continue
            neighbor = GarciaDyadicCell(
                cell.chart_id, axis_depth, neighbor_coordinates
            )
            if neighbor not in active:
                same_chart_boundary.add(cell)
    stored_same_boundary = tuple(token(raw) for raw in payload["same_chart_open_boundary_cells"])
    if stored_same_boundary != tuple(sorted(same_chart_boundary)):
        raise ValueError("orbit-tube same-chart open boundary is inconsistent")
    return GarciaOrbitTubeConstruction(
        family=family,
        normalized_radius=float(radius),
        axis_depth=axis_depth,
        reference_provenance=dict(reference),
        attachment_carriers=tuple(carriers),
        raw_base_cells=int(raw_summary["base"]),
        raw_handle_cells=int(raw_summary["handle"]),
        added_base_attachment_cells=int(payload["added_base_attachment_cells"]),
        active_quotient_incidences=int(payload["active_quotient_incidences"]),
        reverse_open_quotient_incidences=int(reverse_open),
        same_chart_open_boundary_cells=stored_same_boundary,
        reverse_open_base_boundary_cells=stored_reverse_base,
    )


def build_dense_garcia_orbit_tube(
    *,
    axis_depth: int = 5,
    normalized_radius: float = 0.0625,
    gamma: float = DEFAULT_GAMMA,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
    reference_max_step: float = 0.005,
    initial_base_samples_per_stride: int = 257,
    maximum_base_samples_per_stride: int = 16385,
    maximum_handle_samples_per_jump: int = 4097,
) -> GarciaOrbitTubeConstruction:
    """Build a stabilized, nonpercolating fixed-radius depth-20 tube."""

    if axis_depth < 1 or axis_depth > 5:
        raise ValueError("orbit-tube builder currently supports axis depths 1 through 5")
    if not np.isfinite(normalized_radius) or not 0.0 < normalized_radius < 0.5:
        raise ValueError("normalized_radius must lie strictly between zero and one half")
    if initial_base_samples_per_stride < 3:
        raise ValueError("initial dense sampling must contain at least three points")
    if maximum_base_samples_per_stride < initial_base_samples_per_stride:
        raise ValueError("maximum dense sampling is below its initial value")
    if maximum_handle_samples_per_jump < 3:
        raise ValueError("handle sampling must contain at least three points")

    walker = GarciaPassiveWalker(
        gamma=gamma,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
        max_jumps=20,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    charts = garcia_passive_walker_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
    )
    trajectory = walker.system.simulate(
        post_impact_state(*PERIOD_TWO_POINT_A),
        (0.0, 8.0),
        max_jumps=2,
        dense_output=True,
        max_step=reference_max_step,
    )
    if len(trajectory.jump_states) < 2 or len(trajectory.segments) < 2:
        raise RuntimeError("stored Garcia reference did not complete two strides")
    if any(segment.solution is None for segment in trajectory.segments[:2]):
        raise RuntimeError("stored Garcia reference lacks dense segment output")

    convergence: list[dict[str, object]] = []
    previous: tuple[np.ndarray, np.ndarray] | None = None
    base_samples = initial_base_samples_per_stride
    stable = False
    final_points: tuple[np.ndarray, np.ndarray] | None = None
    while base_samples <= maximum_base_samples_per_stride:
        handle_samples = min(
            maximum_handle_samples_per_jump,
            max(3, (base_samples + 1) // 2),
        )
        base_points, handle_points = _dense_reference_points(
            trajectory,
            charts,
            base_samples_per_stride=base_samples,
            handle_samples_per_jump=handle_samples,
        )
        masks = (
            _closed_tube_mask(
                base_points,
                charts.base_bounds,
                axis_depth=axis_depth,
                normalized_radius=normalized_radius,
            ),
            _closed_tube_mask(
                handle_points,
                charts.handle_bounds,
                axis_depth=axis_depth,
                normalized_radius=normalized_radius,
            ),
        )
        changes = (
            None
            if previous is None
            else [
                int(np.count_nonzero(masks[index] ^ previous[index]))
                for index in (0, 1)
            ]
        )
        convergence.append(
            {
                "base_samples_per_stride": base_samples,
                "handle_samples_per_jump": handle_samples,
                "base_points": len(base_points),
                "handle_points": len(handle_points),
                "tube_cells": [int(mask.sum()) for mask in masks],
                "changed_from_previous": changes,
            }
        )
        final_points = base_points, handle_points
        if changes == [0, 0]:
            stable = True
            previous = masks
            break
        previous = masks
        base_samples = 2 * base_samples - 1
    if not stable or previous is None or final_points is None:
        raise RuntimeError(
            "dense reference tube did not stabilize before its declared sampling cap"
        )

    base_mask, handle_mask = previous
    incidence = GarciaWalkerQuotientIncidence(
        charts,
        phi_dot_min=charts.base_bounds[3][0],
        phi_dot_max=charts.base_bounds[3][1],
        transversality_eta=transversality_eta,
    )
    subdivisions = 2**axis_depth
    selected_handle_faces = tuple(
        GarciaDyadicCell(charts.handle_chart_id, axis_depth, tuple(map(int, value)))
        for value in np.argwhere(handle_mask)
        if int(value[-1]) in (0, subdivisions - 1)
    )
    carriers = tuple(
        (handle, _handle_attachment_carrier(handle, charts=charts, incidence=incidence))
        for handle in selected_handle_faces
    )
    raw_base = int(base_mask.sum())
    raw_handle = int(handle_mask.sum())
    for _handle, base_cells in carriers:
        for base in base_cells:
            base_mask[base.coordinates] = True
    added_base = int(base_mask.sum()) - raw_base

    base_tokens = tuple(
        GarciaDyadicCell(charts.base_chart_id, axis_depth, tuple(map(int, value)))
        for value in np.argwhere(base_mask)
    )
    handle_tokens = tuple(
        GarciaDyadicCell(charts.handle_chart_id, axis_depth, tuple(map(int, value)))
        for value in np.argwhere(handle_mask)
    )
    active_base = frozenset(base_tokens)
    active_handle = frozenset(handle_tokens)
    active_incidences = sum(
        base in active_base
        for handle, base_cells in carriers
        if handle in active_handle
        for base in base_cells
    )
    # Count the deliberately omitted reverse carrier: active base cells may
    # meet other, inactive handle cofaces.  Enumerating all boundary cells is
    # cheap at depth 20 (2 * 32^3) and produces an exact open-boundary count.
    reverse_open = 0
    reverse_open_base: set[GarciaDyadicCell] = set()
    for coordinates in itertools.product(range(subdivisions), repeat=3):
        for phase_index in (0, subdivisions - 1):
            handle = GarciaDyadicCell(
                charts.handle_chart_id,
                axis_depth,
                (*coordinates, phase_index),
            )
            if handle in active_handle:
                continue
            for base in _handle_attachment_carrier(
                handle,
                charts=charts,
                incidence=incidence,
            ):
                if base in active_base:
                    reverse_open += 1
                    reverse_open_base.add(base)

    active = active_base | active_handle
    same_chart_boundary: set[GarciaDyadicCell] = set()
    glued_handles = {handle for handle, bases in carriers if bases}
    offsets = tuple(
        offset
        for offset in itertools.product((-1, 0, 1), repeat=4)
        if any(offset)
    )
    for cell in active:
        for offset in offsets:
            neighbor_coordinates = tuple(
                coordinate + delta
                for coordinate, delta in zip(cell.coordinates, offset, strict=True)
            )
            outside_axes = {
                axis
                for axis, coordinate in enumerate(neighbor_coordinates)
                if coordinate < 0 or coordinate >= subdivisions
            }
            if outside_axes:
                if cell in glued_handles and outside_axes == {3}:
                    continue
                same_chart_boundary.add(cell)
                continue
            neighbor = GarciaDyadicCell(
                cell.chart_id, axis_depth, neighbor_coordinates
            )
            if neighbor not in active:
                same_chart_boundary.add(cell)

    trace_digest = hashlib.sha256()
    for points in final_points:
        trace_digest.update(
            np.asarray(points, dtype="<f8").tobytes(order="C")
        )
    reference = {
        "construction": (
            "reference-only dense integration of the stored period-two seed for "
            "two continuous strides plus both intrinsic handle traversals"
        ),
        "gamma": gamma,
        "guard_delta": guard_delta,
        "transversality_eta": transversality_eta,
        "domain_bounds": [list(interval) for interval in walker.domain_bounds],
        "period_two_point_A": list(PERIOD_TWO_POINT_A),
        "period_two_point_B": list(PERIOD_TWO_POINT_B),
        "initial_full_state": post_impact_state(*PERIOD_TWO_POINT_A).tolist(),
        "time_span": [0.0, 8.0],
        "max_jumps_requested": 2,
        "completed_jumps": len(trajectory.jump_states),
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
        "max_step": reference_max_step,
        "dense_output": True,
        "segment_intervals": [
            [float(segment.t_start), float(segment.t_end)]
            for segment in trajectory.segments[:2]
        ],
        "guard_states": [
            np.asarray(guard, dtype=np.float64).tolist()
            for guard, _reset in trajectory.jump_states[:2]
        ],
        "reset_states": [
            np.asarray(reset, dtype=np.float64).tolist()
            for _guard, reset in trajectory.jump_states[:2]
        ],
        "sampling_convergence": convergence,
        "sampling_stable_at_final_doubling": stable,
        "final_base_samples_per_stride": convergence[-1][
            "base_samples_per_stride"
        ],
        "final_handle_samples_per_jump": convergence[-1][
            "handle_samples_per_jump"
        ],
        "dense_coordinate_hash_rule": (
            "SHA-256 of base then handle C-order raw little-endian float64 bytes"
        ),
        "dense_coordinate_sha256": trace_digest.hexdigest(),
    }
    family = GarciaMixedDyadicFamily(
        tuple(sorted((*base_tokens, *handle_tokens))),
        role=(
            "dense two-stride orbit-validation tube; normalized L-infinity radius "
            f"{normalized_radius:g}; one-pass handle-face attachment; open complement"
        ),
    )
    return GarciaOrbitTubeConstruction(
        family=family,
        normalized_radius=float(normalized_radius),
        axis_depth=axis_depth,
        reference_provenance=reference,
        attachment_carriers=carriers,
        raw_base_cells=raw_base,
        raw_handle_cells=raw_handle,
        added_base_attachment_cells=added_base,
        active_quotient_incidences=int(active_incidences),
        reverse_open_quotient_incidences=int(reverse_open),
        same_chart_open_boundary_cells=tuple(sorted(same_chart_boundary)),
        reverse_open_base_boundary_cells=tuple(sorted(reverse_open_base)),
    )


__all__ = [
    "ATTACHMENT_SCHEMA",
    "DEFAULT_TUBE_SHA256",
    "TUBE_SCHEMA",
    "GarciaOrbitTubeConstruction",
    "GarciaOrbitTubeBoundaryAudit",
    "audit_garcia_orbit_tube_candidate_boundary",
    "garcia_orbit_tube_acceptance_blockers",
    "build_dense_garcia_orbit_tube",
    "read_garcia_orbit_tube",
]
