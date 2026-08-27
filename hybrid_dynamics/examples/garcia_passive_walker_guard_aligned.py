"""Guard-aligned local Atlas window for the Garcia passive walker.

The base coordinate change

``(theta, omega, phi, phi_dot) -> (theta, omega, q, nu)``

with ``q=phi-2*theta`` and ``nu=phi_dot-2*omega`` makes the heel-strike guard
the interior cubical hyperplane ``q=0``.  This module builds one predeclared
local validation family.  It is the union of

* a fixed normalized-radius tube around the complete stored two-stride trace;
* one-cell neighborhoods of the fourteen reference boxes rejected by the
  earlier physical-coordinate depth-20 run; and
* one-cell neighborhoods of their persisted missing-image corridors and raw
  target pieces, transformed without looking at a new Morse graph.

Selected handle faces contribute their complete base attachment carrier once.
Reverse base-to-handle cofaces remain an explicit open boundary.  The builder
evaluates only the inexpensive reference trajectory; source-box dynamics and
CMGDB are intentionally separate.
"""

from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .garcia_passive_walker_atlas import (
    AtlasWalkerCell,
    GuardAlignedGarciaWalkerQuotientIncidence,
    garcia_guard_aligned_atlas_charts,
    garcia_passive_walker_atlas_charts,
)
from .garcia_passive_walker_csr import validate_garcia_csr_bundle
from .garcia_passive_walker_local import (
    GarciaDyadicCell,
    GarciaMixedDyadicFamily,
    _dyadic_rectangle_cover,
)
from .garcia_passive_walker_suspension import (
    DEFAULT_GAMMA,
    DEFAULT_GUARD_DELTA,
    DEFAULT_TRANSVERSALITY_ETA,
    PERIOD_TWO_POINT_A,
    PERIOD_TWO_POINT_B,
    GuardAlignedGarciaPassiveWalker,
    guard_aligned_post_impact_state,
)
from .garcia_passive_walker_tube import (
    _closed_tube_mask,
    _dense_reference_points,
    _handle_attachment_carrier,
)


GUARD_ALIGNED_TUBE_SCHEMA: Final = "garcia-guard-aligned-local-tube-v1"
GUARD_ALIGNED_FAMILY_ALGORITHM: Final = (
    "fixed-radius-trace-plus-pinned-unsafe-corridors-one-ring-v1"
)
PINNED_UNSAFE_REFERENCE_SOURCE_IDS: Final = (
    1078,
    2207,
    2751,
    4419,
    5475,
    5782,
    8878,
    14553,
    17212,
    18397,
    23039,
    25071,
    26278,
    28382,
)
PINNED_UNSAFE_SOURCE_IDS_SHA256: Final = (
    "64b2d252433fb107dbe9052b1ee46feb00a11c968e8a04c8fd00247cb44c7db1"
)
PINNED_LEGACY_BUNDLE_FINGERPRINT: Final = (
    "2d5a88134ff2bc426efce0ff9747b45391b7d6fa26c11dfcb621370891bccb18"
)
PINNED_NORMALIZED_RADIUS: Final = 0.0625
PINNED_REFERENCE_MAX_STEP: Final = 0.005
PINNED_INITIAL_BASE_SAMPLES_PER_STRIDE: Final = 257
PINNED_MAXIMUM_BASE_SAMPLES_PER_STRIDE: Final = 16385
PINNED_MAXIMUM_HANDLE_SAMPLES_PER_JUMP: Final = 4097


def default_legacy_tube_bundle() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "garcia_passive_walker_atlas"
        / "tube_relation_bundle_tau050_depth20_geom8b7625700202"
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _token_payload(cell: GarciaDyadicCell) -> dict[str, object]:
    return {
        "chart_id": cell.chart_id,
        "axis_depth": cell.axis_depth,
        "coordinates": list(cell.coordinates),
    }


def _transform_physical_rectangle(flat_bounds: tuple[float, ...]) -> tuple[float, ...]:
    values = np.asarray(flat_bounds, dtype=np.float64)
    if values.shape != (8,) or not np.all(np.isfinite(values)):
        raise ValueError("physical Garcia rectangle must have eight finite bounds")
    lower, upper = values[:4], values[4:]
    if np.any(lower > upper):
        raise ValueError("physical Garcia rectangle has reversed bounds")
    return (
        float(lower[0]),
        float(lower[1]),
        float(lower[2] - 2.0 * upper[0]),
        float(lower[3] - 2.0 * upper[1]),
        float(upper[0]),
        float(upper[1]),
        float(upper[2] - 2.0 * lower[0]),
        float(upper[3] - 2.0 * lower[1]),
    )


def _transform_intrinsic_handle_rectangle(
    flat_bounds: tuple[float, ...],
    *,
    phi_dot_min: float,
    phi_dot_max: float,
    transversality_eta: float,
) -> tuple[float, ...]:
    """Transform old ``(theta,omega,rho,s)`` bounds to ``(theta,omega,nu,s)``.

    The old parametrization is piecewise affine in omega, with its only kink
    at ``(phi_dot_min-eta)/2``.  Evaluating both omega/rho endpoints and that
    kink gives the exact interval hull for ``nu``.
    """

    values = np.asarray(flat_bounds, dtype=np.float64)
    if values.shape != (8,) or not np.all(np.isfinite(values)):
        raise ValueError("intrinsic Garcia handle rectangle is malformed")
    lower, upper = values[:4], values[4:]
    if np.any(lower > upper):
        raise ValueError("intrinsic Garcia handle rectangle has reversed bounds")
    kink = 0.5 * (float(phi_dot_min) - float(transversality_eta))
    omega_nodes = {float(lower[1]), float(upper[1])}
    if lower[1] <= kink <= upper[1]:
        omega_nodes.add(kink)
    nu_values = []
    for omega in sorted(omega_nodes):
        minimum = max(float(phi_dot_min), 2.0 * omega + transversality_eta)
        for rho in (float(lower[2]), float(upper[2])):
            phi_dot = minimum + rho * (float(phi_dot_max) - minimum)
            nu_values.append(phi_dot - 2.0 * omega)
    return (
        float(lower[0]),
        float(lower[1]),
        float(min(nu_values)),
        float(lower[3]),
        float(upper[0]),
        float(upper[1]),
        float(max(nu_values)),
        float(upper[3]),
    )


def _transform_legacy_rectangle(
    chart_id: int,
    flat_bounds: tuple[float, ...],
    *,
    phi_dot_min: float,
    phi_dot_max: float,
    transversality_eta: float,
) -> tuple[float, ...]:
    if chart_id == 0:
        return _transform_physical_rectangle(flat_bounds)
    if chart_id == 1:
        return _transform_intrinsic_handle_rectangle(
            flat_bounds,
            phi_dot_min=phi_dot_min,
            phi_dot_max=phi_dot_max,
            transversality_eta=transversality_eta,
        )
    raise ValueError(f"unknown legacy Garcia chart {chart_id}")


def _pinned_legacy_records(
    bundle_path: str | Path,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    provenance_path, manifest = validate_garcia_csr_bundle(bundle_path)
    if manifest.get("fingerprint") != PINNED_LEGACY_BUNDLE_FINGERPRINT:
        raise ValueError(
            "legacy diagnostic bundle is not the predeclared depth-20 tube relation"
        )
    wanted = set(PINNED_UNSAFE_REFERENCE_SOURCE_IDS)
    records: dict[int, dict[str, object]] = {}
    with gzip.open(provenance_path, "rt", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        for line in stream:
            record = json.loads(line)
            if record.get("record") == "cell" and record.get("index") in wanted:
                records[int(record["index"])] = record
    if set(records) != wanted:
        raise ValueError("pinned legacy bundle lacks one of the fourteen source records")
    identifier_digest = hashlib.sha256(
        _canonical_json(list(PINNED_UNSAFE_REFERENCE_SOURCE_IDS))
    ).hexdigest()
    if identifier_digest != PINNED_UNSAFE_SOURCE_IDS_SHA256:
        raise RuntimeError("pinned unsafe-source identifier digest is inconsistent")
    configuration = header.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("legacy CSR provenance configuration is malformed")
    if (
        configuration.get("model")
        != "garcia_passive_walker_fixed_time_suspension_atlas"
        or configuration.get("physical_model_revision")
        != "garcia-physical-event-reset-v2"
        or configuration.get("boxmap_revision")
        != "garcia-fixed-time-suspension-sample-bloat-v2"
        or configuration.get("t_star") != 0.5
        or configuration.get("gamma") != DEFAULT_GAMMA
        or configuration.get("guard_delta") != DEFAULT_GUARD_DELTA
        or configuration.get("transversality_eta")
        != DEFAULT_TRANSVERSALITY_ETA
        or configuration.get("samples_per_axis") != 3
        or configuration.get("padding_cells") != 1.0
        or configuration.get("max_step") != 0.02
        or configuration.get("max_jumps") != 20
        or configuration.get("require_domain_path") is not True
        or configuration.get("min_axis_depth") != 5
        or configuration.get("max_axis_depth") != 5
    ):
        raise ValueError("legacy unsafe-source provenance has the wrong pinned run")
    return (
        {
            "bundle_schema": manifest["schema"],
            "bundle_fingerprint": manifest["fingerprint"],
            "relation_fingerprint": manifest["relation_fingerprint"],
            "provenance_sha256": manifest["provenance_sha256"],
            "configuration": configuration,
            "unsafe_source_ids": list(PINNED_UNSAFE_REFERENCE_SOURCE_IDS),
            "unsafe_source_ids_sha256": identifier_digest,
        },
        tuple(records[index] for index in PINNED_UNSAFE_REFERENCE_SOURCE_IDS),
    )


def _closed_one_ring(
    cells: set[GarciaDyadicCell],
    *,
    axis_depth: int,
) -> set[GarciaDyadicCell]:
    subdivisions = 2**axis_depth
    result = set(cells)
    for cell in cells:
        for delta in itertools.product((-1, 0, 1), repeat=4):
            coordinates = tuple(
                value + offset
                for value, offset in zip(cell.coordinates, delta, strict=True)
            )
            if all(0 <= value < subdivisions for value in coordinates):
                result.add(GarciaDyadicCell(cell.chart_id, axis_depth, coordinates))
    return result


@dataclass(frozen=True)
class GuardAlignedGarciaTubeConstruction:
    family: GarciaMixedDyadicFamily
    axis_depth: int
    normalized_radius: float
    charts: object
    reference_provenance: dict[str, object]
    legacy_provenance: dict[str, object]
    attachment_carriers: tuple[
        tuple[GarciaDyadicCell, tuple[GarciaDyadicCell, ...]], ...
    ]
    raw_base_tube_cells: int
    raw_handle_tube_cells: int
    transformed_source_cells: int
    transformed_missing_corridor_cells: int
    transformed_target_piece_cells: int
    enrichment_one_ring_cells: int
    added_base_attachment_cells: int
    active_quotient_incidences: int
    reverse_open_quotient_incidences: int
    same_chart_open_boundary_cells: tuple[GarciaDyadicCell, ...]
    reverse_open_base_boundary_cells: tuple[GarciaDyadicCell, ...]

    def to_dict(self) -> dict[str, object]:
        cells = [_token_payload(cell) for cell in self.family.cells]
        carriers = [
            {
                "handle": {
                    **_token_payload(handle),
                    "face": "guard_s0" if handle.coordinates[-1] == 0 else "reset_s1",
                },
                "base": [_token_payload(base) for base in bases],
            }
            for handle, bases in self.attachment_carriers
        ]
        payload: dict[str, object] = {
            "schema": GUARD_ALIGNED_TUBE_SCHEMA,
            "algorithm": GUARD_ALIGNED_FAMILY_ALGORITHM,
            "construction_role": "predeclared_local_validation_not_discovery",
            "coordinate_system": "guard_aligned_q_nu",
            "scientific_result_accepted": False,
            "whole_cell_outer_enclosure_certified": False,
            "source_box_dynamics_evaluated": False,
            "reference_only_simulation_evaluated": True,
            "axis_depth": self.axis_depth,
            "total_depth": 4 * self.axis_depth,
            "normalized_Linf_radius": self.normalized_radius,
            "attachment_policy": (
                "selected_handle_faces_to_complete_internal_q0_base_carriers_once; "
                "reverse cofaces open"
            ),
            "active_cells": len(self.family.cells),
            "active_chart_counts": self.family.chart_counts(),
            "raw_trace_tube_cells": {
                "base": self.raw_base_tube_cells,
                "handle": self.raw_handle_tube_cells,
            },
            "fixed_prior_diagnostic_enrichment": {
                "transformed_source_cells_before_ring": self.transformed_source_cells,
                "transformed_missing_corridor_cells_before_ring": (
                    self.transformed_missing_corridor_cells
                ),
                "transformed_target_piece_cells_before_ring": (
                    self.transformed_target_piece_cells
                ),
                "union_after_one_closed_cell_ring": self.enrichment_one_ring_cells,
                "selection_uses_new_relation_or_SCC_counts": False,
            },
            "added_base_attachment_cells": self.added_base_attachment_cells,
            "active_quotient_incidences": self.active_quotient_incidences,
            "reverse_open_quotient_incidences": self.reverse_open_quotient_incidences,
            "same_chart_open_boundary_cells": [
                _token_payload(cell) for cell in self.same_chart_open_boundary_cells
            ],
            "reverse_open_base_boundary_cells": [
                _token_payload(cell) for cell in self.reverse_open_base_boundary_cells
            ],
            "attachment_carriers": carriers,
            "attachment_carriers_sha256": hashlib.sha256(
                _canonical_json(carriers)
            ).hexdigest(),
            "reference_trace": self.reference_provenance,
            "legacy_negative_diagnostic": self.legacy_provenance,
            "family_cells_sha256": hashlib.sha256(_canonical_json(cells)).hexdigest(),
            "cells": cells,
        }
        payload["fingerprint"] = {
            "algorithm": "sha256",
            "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        }
        return payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(self.to_dict(), sort_keys=True, allow_nan=False) + "\n"
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            if target.suffix == ".gz":
                with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                    stream.write(encoded)
            else:
                temporary.write_text(encoded, encoding="utf-8")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target


def _same_chart_open_boundary(
    active: set[GarciaDyadicCell],
    glued_handles: set[GarciaDyadicCell],
    *,
    axis_depth: int,
) -> set[GarciaDyadicCell]:
    subdivisions = 2**axis_depth
    result: set[GarciaDyadicCell] = set()
    offsets = tuple(
        delta
        for delta in itertools.product((-1, 0, 1), repeat=4)
        if any(delta)
    )
    for cell in active:
        for delta in offsets:
            coordinates = tuple(
                value + offset
                for value, offset in zip(cell.coordinates, delta, strict=True)
            )
            outside = {
                axis
                for axis, value in enumerate(coordinates)
                if value < 0 or value >= subdivisions
            }
            if outside:
                if cell in glued_handles and outside == {3}:
                    continue
                result.add(cell)
                break
            if GarciaDyadicCell(cell.chart_id, axis_depth, coordinates) not in active:
                result.add(cell)
                break
    return result


def build_guard_aligned_garcia_tube(
    *,
    axis_depth: int = 5,
    normalized_radius: float = 0.0625,
    legacy_bundle_path: str | Path | None = None,
    gamma: float = DEFAULT_GAMMA,
    guard_delta: float = DEFAULT_GUARD_DELTA,
    transversality_eta: float = DEFAULT_TRANSVERSALITY_ETA,
    reference_max_step: float = 0.005,
    initial_base_samples_per_stride: int = 257,
    maximum_base_samples_per_stride: int = 16385,
    maximum_handle_samples_per_jump: int = 4097,
) -> GuardAlignedGarciaTubeConstruction:
    """Build the fixed trace-plus-prior-diagnostic aligned local family."""

    if axis_depth < 1 or axis_depth > 5:
        raise ValueError("guard-aligned tube supports axis depths one through five")
    pinned_arguments = {
        "normalized_radius": (normalized_radius, PINNED_NORMALIZED_RADIUS),
        "gamma": (gamma, DEFAULT_GAMMA),
        "guard_delta": (guard_delta, DEFAULT_GUARD_DELTA),
        "transversality_eta": (
            transversality_eta,
            DEFAULT_TRANSVERSALITY_ETA,
        ),
        "reference_max_step": (reference_max_step, PINNED_REFERENCE_MAX_STEP),
        "initial_base_samples_per_stride": (
            initial_base_samples_per_stride,
            PINNED_INITIAL_BASE_SAMPLES_PER_STRIDE,
        ),
        "maximum_base_samples_per_stride": (
            maximum_base_samples_per_stride,
            PINNED_MAXIMUM_BASE_SAMPLES_PER_STRIDE,
        ),
        "maximum_handle_samples_per_jump": (
            maximum_handle_samples_per_jump,
            PINNED_MAXIMUM_HANDLE_SAMPLES_PER_JUMP,
        ),
    }
    mismatches = [
        name for name, (actual, expected) in pinned_arguments.items() if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "the flagship guard-aligned family has fixed predeclared settings; "
            f"mismatched: {', '.join(mismatches)}"
        )
    legacy_bundle = default_legacy_tube_bundle() if legacy_bundle_path is None else Path(legacy_bundle_path)
    legacy_provenance, legacy_records = _pinned_legacy_records(legacy_bundle)
    legacy_configuration = legacy_provenance["configuration"]
    physical_bounds = legacy_configuration["base_bounds"]
    phi_dot_min, phi_dot_max = map(float, physical_bounds[3])
    old_charts = garcia_passive_walker_atlas_charts(
        base_bounds=physical_bounds,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
    )

    walker = GuardAlignedGarciaPassiveWalker(
        gamma=gamma,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
        max_jumps=20,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    charts = garcia_guard_aligned_atlas_charts(
        base_bounds=walker.domain_bounds,
        guard_delta=guard_delta,
        transversality_eta=transversality_eta,
    )
    trajectory = walker.system.simulate(
        guard_aligned_post_impact_state(*PERIOD_TWO_POINT_A),
        (0.0, 8.0),
        max_jumps=2,
        dense_output=True,
        max_step=reference_max_step,
    )
    if len(trajectory.jump_states) < 2 or len(trajectory.segments) < 2:
        raise RuntimeError("stored aligned Garcia reference did not complete two strides")

    convergence: list[dict[str, object]] = []
    previous: tuple[np.ndarray, np.ndarray] | None = None
    final_points: tuple[np.ndarray, np.ndarray] | None = None
    samples = int(initial_base_samples_per_stride)
    while samples <= maximum_base_samples_per_stride:
        handle_samples = min(
            maximum_handle_samples_per_jump,
            max(3, (samples + 1) // 2),
        )
        points = _dense_reference_points(
            trajectory,
            charts,
            base_samples_per_stride=samples,
            handle_samples_per_jump=handle_samples,
        )
        masks = (
            _closed_tube_mask(
                points[0],
                charts.base_bounds,
                axis_depth=axis_depth,
                normalized_radius=normalized_radius,
            ),
            _closed_tube_mask(
                points[1],
                charts.handle_bounds,
                axis_depth=axis_depth,
                normalized_radius=normalized_radius,
            ),
        )
        changes = (
            None
            if previous is None
            else [int(np.count_nonzero(a ^ b)) for a, b in zip(masks, previous)]
        )
        convergence.append(
            {
                "base_samples_per_stride": samples,
                "handle_samples_per_jump": handle_samples,
                "tube_cells": [int(mask.sum()) for mask in masks],
                "changed_from_previous": changes,
            }
        )
        final_points = points
        previous = masks
        if changes == [0, 0]:
            break
        samples = 2 * samples - 1
    if previous is None or final_points is None or convergence[-1]["changed_from_previous"] != [0, 0]:
        raise RuntimeError("guard-aligned trace tube did not stabilize at the declared cap")

    base_mask, handle_mask = previous
    raw_base = int(base_mask.sum())
    raw_handle = int(handle_mask.sum())
    source_cells: set[GarciaDyadicCell] = set()
    missing_cells: set[GarciaDyadicCell] = set()
    target_cells: set[GarciaDyadicCell] = set()

    def transformed_cover(chart_id: int, bounds: tuple[float, ...]) -> set[GarciaDyadicCell]:
        transformed = _transform_legacy_rectangle(
            chart_id,
            bounds,
            phi_dot_min=phi_dot_min,
            phi_dot_max=phi_dot_max,
            transversality_eta=transversality_eta,
        )
        return set(
            _dyadic_rectangle_cover(charts, chart_id, transformed, axis_depth)[0]
        )

    for record in legacy_records:
        chart_id = int(record["chart_id"])
        source_cells.update(
            transformed_cover(chart_id, tuple(float(v) for v in record["bounds"]))
        )
        open_exit = record["open_exit"]
        for raw in open_exit["missing_in_domain_witnesses"]:
            token = GarciaDyadicCell(
                int(raw["chart_id"]),
                int(raw["axis_depth"]),
                tuple(int(value) for value in raw["coordinates"]),
            )
            missing_cells.update(
                transformed_cover(token.chart_id, token.bounds(old_charts))
            )
        for piece in open_exit["target_pieces"]:
            target_cells.update(
                transformed_cover(
                    int(piece["chart_id"]),
                    tuple(float(value) for value in piece["bounds"]),
                )
            )
    enrichment = _closed_one_ring(
        source_cells | missing_cells | target_cells,
        axis_depth=axis_depth,
    )
    for cell in enrichment:
        (base_mask if cell.chart_id == charts.base_chart_id else handle_mask)[
            cell.coordinates
        ] = True

    incidence = GuardAlignedGarciaWalkerQuotientIncidence(
        charts,
        transversality_eta=transversality_eta,
    )
    subdivisions = 2**axis_depth
    selected_faces = tuple(
        GarciaDyadicCell(charts.handle_chart_id, axis_depth, tuple(map(int, value)))
        for value in np.argwhere(handle_mask)
        if int(value[-1]) in (0, subdivisions - 1)
    )
    carriers = tuple(
        (
            handle,
            _handle_attachment_carrier(handle, charts=charts, incidence=incidence),
        )
        for handle in selected_faces
    )
    base_before_attachment = int(base_mask.sum())
    for _handle, bases in carriers:
        for base in bases:
            base_mask[base.coordinates] = True
    added_base = int(base_mask.sum()) - base_before_attachment

    base_tokens = tuple(
        GarciaDyadicCell(charts.base_chart_id, axis_depth, tuple(map(int, value)))
        for value in np.argwhere(base_mask)
    )
    handle_tokens = tuple(
        GarciaDyadicCell(charts.handle_chart_id, axis_depth, tuple(map(int, value)))
        for value in np.argwhere(handle_mask)
    )
    active_base, active_handle = frozenset(base_tokens), frozenset(handle_tokens)
    active_incidences = sum(
        base in active_base
        for handle, bases in carriers
        if handle in active_handle
        for base in bases
    )
    reverse_open = 0
    reverse_base: set[GarciaDyadicCell] = set()
    for coordinates in itertools.product(range(subdivisions), repeat=3):
        for phase in (0, subdivisions - 1):
            handle = GarciaDyadicCell(
                charts.handle_chart_id,
                axis_depth,
                (*coordinates, phase),
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
                    reverse_base.add(base)

    active = set(active_base | active_handle)
    glued = {handle for handle, bases in carriers if bases}
    same_boundary = _same_chart_open_boundary(
        active,
        glued,
        axis_depth=axis_depth,
    )
    trace_digest = hashlib.sha256()
    for points in final_points:
        trace_digest.update(np.asarray(points, dtype="<f8").tobytes(order="C"))
    reference = {
        "coordinate_system": "theta_omega_q_nu",
        "gamma": gamma,
        "guard_delta": guard_delta,
        "transversality_eta": transversality_eta,
        "domain_bounds": [list(interval) for interval in walker.domain_bounds],
        "period_two_point_A": list(PERIOD_TWO_POINT_A),
        "period_two_point_B": list(PERIOD_TWO_POINT_B),
        "initial_state": guard_aligned_post_impact_state(*PERIOD_TWO_POINT_A).tolist(),
        "reference_max_step": reference_max_step,
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
        "sampling_convergence": convergence,
        "sampling_stable_at_final_doubling": True,
        "dense_coordinate_sha256": trace_digest.hexdigest(),
    }
    family = GarciaMixedDyadicFamily(
        tuple(sorted(active)),
        role=(
            "guard-aligned predeclared local validation; fixed radius 0.0625; "
            "pinned fourteen-source missing-corridor one-ring; open complement"
        ),
    )
    return GuardAlignedGarciaTubeConstruction(
        family=family,
        axis_depth=axis_depth,
        normalized_radius=float(normalized_radius),
        charts=charts,
        reference_provenance=reference,
        legacy_provenance=legacy_provenance,
        attachment_carriers=carriers,
        raw_base_tube_cells=raw_base,
        raw_handle_tube_cells=raw_handle,
        transformed_source_cells=len(source_cells),
        transformed_missing_corridor_cells=len(missing_cells),
        transformed_target_piece_cells=len(target_cells),
        enrichment_one_ring_cells=len(enrichment),
        added_base_attachment_cells=added_base,
        active_quotient_incidences=int(active_incidences),
        reverse_open_quotient_incidences=int(reverse_open),
        same_chart_open_boundary_cells=tuple(sorted(same_boundary)),
        reverse_open_base_boundary_cells=tuple(sorted(reverse_base)),
    )


__all__ = [
    "GUARD_ALIGNED_FAMILY_ALGORITHM",
    "GUARD_ALIGNED_TUBE_SCHEMA",
    "PINNED_UNSAFE_REFERENCE_SOURCE_IDS",
    "PINNED_UNSAFE_SOURCE_IDS_SHA256",
    "PINNED_LEGACY_BUNDLE_FINGERPRINT",
    "PINNED_NORMALIZED_RADIUS",
    "GuardAlignedGarciaTubeConstruction",
    "build_guard_aligned_garcia_tube",
    "default_legacy_tube_bundle",
]
