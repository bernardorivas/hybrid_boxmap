"""Tagged CMGDB box map for a fixed-time hybrid suspension.

The ordinary :mod:`CMGDB` ``Model`` asks its callback for one Euclidean
rectangle.  That is the wrong geometry at a reset: a fixed-time image can
have pieces in the base chart and in the reset-handle chart, and taking their
Euclidean hull invents states between those pieces.  ``CMGDB.AtlasModel``
instead accepts a finite union of tagged rectangles.  The callable defined
here implements exactly that callback interface::

    (source_chart_id, source_bounds) -> [
        (target_chart_id, target_bounds), ...
    ]

Bounds use CMGDB's flattened convention ``[lower..., upper...]``.  The base
chart stores physical states.  The handle chart stores intrinsic guard
coordinates followed by the unit phase coordinate ``s``.

This remains a sampled CMGDB-style box map.  Interior tensor samples and
event-stratum separation make it substantially harder to miss a reset branch
than corner sampling, but they do *not* prove the whole-cell outer-enclosure
condition.  The diagnostics below are audit gates and falsification evidence,
not a certification theorem.

An optional single-handle bridge handles one narrowly scoped sampling defect:
adjacent source tensor nodes can end in ``base(j)`` and ``base(j+1)`` without
sampling the intervening terminal handle.  Deterministic bisection may locate
that handle and add its tagged carrier to the terminal image.  It never inserts
intermediate-time graph edges, never bridges a gap larger than two stages, and
is disabled by default.  This is still a provenance-bearing sampled-cover
assumption, not a rigorous enclosure theorem.
"""

from __future__ import annotations

import itertools
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Hashable

import numpy as np
import numpy.typing as npt

from .hybrid_system import HybridSystem
from .sampled_suspension import (
    BaseSuspensionSample,
    HandleSuspensionSample,
    SuspensionSample,
    simulate_suspension_endpoint,
)


State = npt.NDArray[np.float64]
GuardCoordinates = Callable[[State], npt.ArrayLike]
GuardEmbedding = Callable[[State], npt.ArrayLike]
TaggedRectangle = tuple[int, list[float]]


SINGLE_HANDLE_BRIDGE_ALGORITHM = "terminal-single-handle-bisection-v1"
SINGLE_HANDLE_BRIDGE_ASSUMPTIONS = (
    "the source segment stays inside the rectangular source cell",
    "a base-stage gap of two crosses exactly one unit suspension handle",
    "quotient continuity connects the two base branches through that handle",
    "the emitted sample-and-bloat carrier is a nonrigorous outer-cover assumption",
)


@dataclass(frozen=True)
class SuspensionAtlasCharts:
    """Coordinate data for the base and reset-handle Atlas charts.

    ``base_bounds`` contains one physical interval per state coordinate.
    ``guard_bounds`` contains intrinsic guard-coordinate intervals.  The
    handle chart is ``guard_bounds x [0, 1]`` and therefore normally has the
    same dimension as the base chart for a codimension-one guard.

    ``guard_coordinates`` maps a physical guard state to its intrinsic
    coordinates; ``guard_embedding`` performs the inverse parametrization.
    They need only be mutually consistent on the declared guard patch.
    """

    base_bounds: tuple[tuple[float, float], ...]
    guard_bounds: tuple[tuple[float, float], ...]
    guard_coordinates: GuardCoordinates
    guard_embedding: GuardEmbedding
    base_chart_id: int = 0
    handle_chart_id: int = 1
    base_periodic: tuple[bool, ...] = ()
    handle_periodic: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.base_chart_id, bool) or self.base_chart_id < 0:
            raise ValueError("base_chart_id must be a non-negative integer")
        if isinstance(self.handle_chart_id, bool) or self.handle_chart_id < 0:
            raise ValueError("handle_chart_id must be a non-negative integer")
        if self.base_chart_id == self.handle_chart_id:
            raise ValueError("base and handle chart ids must be distinct")
        if not self.base_bounds:
            raise ValueError("base_bounds must be nonempty")
        _validate_intervals(self.base_bounds, name="base_bounds")
        _validate_intervals(self.guard_bounds, name="guard_bounds")
        if self.handle_dimension != self.base_dimension:
            raise ValueError(
                "the reset handle must have the base dimension: "
                "len(guard_bounds) + 1 == len(base_bounds)"
            )
        if self.base_periodic and len(self.base_periodic) != self.base_dimension:
            raise ValueError("base_periodic must have one flag per base coordinate")
        if self.handle_periodic and len(self.handle_periodic) != self.handle_dimension:
            raise ValueError("handle_periodic must have one flag per handle coordinate")

    @property
    def base_dimension(self) -> int:
        return len(self.base_bounds)

    @property
    def handle_dimension(self) -> int:
        return len(self.guard_bounds) + 1

    @property
    def handle_bounds(self) -> tuple[tuple[float, float], ...]:
        return self.guard_bounds + ((0.0, 1.0),)

    def encode_base(self, state: npt.ArrayLike) -> tuple[float, ...]:
        """Return base-chart coordinates for a physical state."""

        values = np.asarray(state, dtype=np.float64)
        if values.shape != (self.base_dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("base state has the wrong dimension or is non-finite")
        return tuple(float(value) for value in values)

    def decode_base(self, coordinates: npt.ArrayLike) -> State:
        """Return a physical state from base-chart coordinates."""

        return np.asarray(self.encode_base(coordinates), dtype=np.float64)

    def encode_handle(
        self,
        guard_state: npt.ArrayLike,
        phase: float,
    ) -> tuple[float, ...]:
        """Encode a guard state and phase in intrinsic handle coordinates."""

        guard = np.asarray(guard_state, dtype=np.float64)
        if guard.shape != (self.base_dimension,) or not np.all(np.isfinite(guard)):
            raise ValueError("guard state has the wrong dimension or is non-finite")
        phase = float(phase)
        if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
            raise ValueError("handle phase must lie in [0, 1]")
        intrinsic = np.asarray(self.guard_coordinates(guard), dtype=np.float64)
        if intrinsic.shape != (len(self.guard_bounds),) or not np.all(
            np.isfinite(intrinsic)
        ):
            raise ValueError(
                "guard_coordinates returned the wrong dimension or a non-finite value"
            )
        return tuple(float(value) for value in np.concatenate((intrinsic, [phase])))

    def decode_handle(
        self,
        coordinates: npt.ArrayLike,
    ) -> tuple[State, float]:
        """Decode handle coordinates as ``(guard_state, phase)``."""

        values = np.asarray(coordinates, dtype=np.float64)
        if values.shape != (self.handle_dimension,) or not np.all(np.isfinite(values)):
            raise ValueError("handle coordinates have the wrong dimension or are non-finite")
        phase = float(values[-1])
        if not 0.0 <= phase <= 1.0:
            raise ValueError("handle phase must lie in [0, 1]")
        guard = np.asarray(self.guard_embedding(values[:-1]), dtype=np.float64)
        if guard.shape != (self.base_dimension,) or not np.all(np.isfinite(guard)):
            raise ValueError(
                "guard_embedding returned the wrong dimension or a non-finite state"
            )
        return guard, phase

    def bounds_for(self, chart_id: int) -> tuple[tuple[float, float], ...]:
        if chart_id == self.base_chart_id:
            return self.base_bounds
        if chart_id == self.handle_chart_id:
            return self.handle_bounds
        raise ValueError(f"unknown suspension chart id {chart_id}")

    def periodic_for(self, chart_id: int) -> tuple[bool, ...]:
        if chart_id == self.base_chart_id:
            return self.base_periodic or (False,) * self.base_dimension
        if chart_id == self.handle_chart_id:
            return self.handle_periodic or (False,) * self.handle_dimension
        raise ValueError(f"unknown suspension chart id {chart_id}")


@dataclass(frozen=True, order=True)
class SuspensionStratum:
    """One target chart and one suspension event stage.

    Stages are numbered so ``base(j)`` has stage ``2*j`` and ``handle(j)``
    has stage ``2*j + 1``.  Consecutive stages meet at exactly one quotient
    face; a larger gap between adjacent source samples is unresolved.
    """

    chart_id: int
    stage: int
    kind: str


@dataclass(frozen=True)
class SamplingFailure:
    source_point: tuple[float, ...]
    reason: str


@dataclass(frozen=True)
class SingleHandleBridgeProbe:
    """One deterministic bisection probe used to locate a skipped handle."""

    bisection_depth: int
    source_point: tuple[float, ...]
    stage: int | None
    kind: str | None
    failure: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "bisection_depth": self.bisection_depth,
            "source_point": list(self.source_point),
            "stage": self.stage,
            "kind": self.kind,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class SingleHandleBridgeDiagnostic:
    """Authenticated-ready provenance for one skipped terminal handle.

    A successful record describes one *terminal image carrier*.  It never
    denotes an intermediate-time transition of the fixed-time graph.
    """

    algorithm_revision: str
    status: str
    incident_grid_indices: tuple[tuple[int, ...], tuple[int, ...]]
    incident_source_points: tuple[tuple[float, ...], tuple[float, ...]]
    lower_stage: int
    skipped_handle_stage: int
    upper_stage: int
    max_bisections: int
    probes: tuple[SingleHandleBridgeProbe, ...]
    witness_source_point: tuple[float, ...] | None
    guard_state: tuple[float, ...] | None
    reset_state: tuple[float, ...] | None
    witness_phase: float | None
    raw_handle_carrier_bounds: tuple[float, ...] | None
    emitted_handle_carrier_bounds: tuple[float, ...] | None
    assumptions: tuple[str, ...] = SINGLE_HANDLE_BRIDGE_ASSUMPTIONS

    @property
    def synthesized(self) -> bool:
        return self.status == "synthesized"

    @property
    def failed_probe_count(self) -> int:
        return sum(probe.failure is not None for probe in self.probes)

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_revision": self.algorithm_revision,
            "status": self.status,
            "synthesized": self.synthesized,
            "terminal_image_carrier_only": True,
            "intermediate_time_graph_edges_added": False,
            "incident_grid_indices": [
                list(index) for index in self.incident_grid_indices
            ],
            "incident_source_points": [
                list(point) for point in self.incident_source_points
            ],
            "lower_stage": self.lower_stage,
            "skipped_handle_stage": self.skipped_handle_stage,
            "upper_stage": self.upper_stage,
            "max_bisections": self.max_bisections,
            "probes": [probe.to_dict() for probe in self.probes],
            "witness_source_point": (
                None
                if self.witness_source_point is None
                else list(self.witness_source_point)
            ),
            "guard_state": (
                None if self.guard_state is None else list(self.guard_state)
            ),
            "reset_state": (
                None if self.reset_state is None else list(self.reset_state)
            ),
            "witness_phase": self.witness_phase,
            "raw_handle_carrier_bounds": (
                None
                if self.raw_handle_carrier_bounds is None
                else list(self.raw_handle_carrier_bounds)
            ),
            "emitted_handle_carrier_bounds": (
                None
                if self.emitted_handle_carrier_bounds is None
                else list(self.emitted_handle_carrier_bounds)
            ),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class SourceBoxMapDiagnostic:
    """Audit record for one invocation of the Atlas box-map callback."""

    source_chart_id: int
    source_bounds: tuple[float, ...]
    tensor_shape: tuple[int, ...]
    sample_count: int
    successful_samples: int
    corner_samples: int
    strata: tuple[SuspensionStratum, ...]
    interior_only_strata: tuple[SuspensionStratum, ...]
    guard_face_witnesses: int
    reset_face_witnesses: int
    raw_unresolved_stage_edges: tuple[tuple[int, int], ...]
    unresolved_stage_edges: tuple[tuple[int, int], ...]
    single_handle_bridge_attempts: tuple[SingleHandleBridgeDiagnostic, ...]
    failures: tuple[SamplingFailure, ...]
    returned_pieces: int
    empty_image: bool
    whole_cell_outer_enclosure_certified: bool = False

    @property
    def passed_sampled_event_gate(self) -> bool:
        """Whether tensor samples succeeded and no stage gap remains unresolved."""

        return not self.failures and not self.unresolved_stage_edges


@dataclass(frozen=True)
class SuspensionBoxMapDiagnostics:
    """Aggregate immutable snapshot of callback diagnostics."""

    single_handle_bridge_enabled: bool
    single_handle_bridge_algorithm_revision: str
    single_handle_bridge_max_bisections: int
    single_handle_bridge_assumptions: tuple[str, ...]
    source_boxes: int
    sampled_points: int
    successful_samples: int
    failed_samples: int
    returned_pieces: int
    empty_images: int
    guard_face_witnesses: int
    reset_face_witnesses: int
    raw_unresolved_stage_edges: int
    unresolved_stage_edges: int
    single_handle_bridge_attempts: int
    synthesized_single_handle_bridges: int
    single_handle_bridge_probe_points: int
    single_handle_bridge_failed_probes: int
    interior_only_strata: int
    retained_source_records: tuple[SourceBoxMapDiagnostic, ...]
    single_handle_bridge_records: tuple[SourceBoxMapDiagnostic, ...]
    whole_cell_outer_enclosure_certified: bool = False


@dataclass(frozen=True)
class _EndpointObservation:
    grid_index: tuple[int, ...]
    source_point: tuple[float, ...]
    corner: bool
    sample: SuspensionSample
    stratum: SuspensionStratum
    target_coordinates: tuple[float, ...]


@dataclass(frozen=True)
class _FaceWitness:
    face: str
    guard_state: tuple[float, ...]
    reset_state: tuple[float, ...]
    jump_index: int
    incident_indices: frozenset[tuple[int, ...]] = frozenset()


@dataclass
class _MutableDiagnostics:
    source_boxes: int = 0
    sampled_points: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    returned_pieces: int = 0
    empty_images: int = 0
    guard_face_witnesses: int = 0
    reset_face_witnesses: int = 0
    raw_unresolved_stage_edges: int = 0
    unresolved_stage_edges: int = 0
    single_handle_bridge_attempts: int = 0
    synthesized_single_handle_bridges: int = 0
    single_handle_bridge_probe_points: int = 0
    single_handle_bridge_failed_probes: int = 0
    interior_only_strata: int = 0
    records: list[SourceBoxMapDiagnostic] = field(default_factory=list)
    bridge_records: list[SourceBoxMapDiagnostic] = field(default_factory=list)


def _validate_intervals(
    intervals: Sequence[Sequence[float]],
    *,
    name: str,
) -> None:
    for axis, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError(f"{name}[{axis}] must contain lower and upper bounds")
        lower, upper = (float(value) for value in interval)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"{name}[{axis}] must be finite")
        if lower >= upper:
            raise ValueError(f"{name}[{axis}] must satisfy lower < upper")


def _split_bounds(
    bounds: Sequence[float],
    dimension: int,
) -> tuple[State, State]:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (2 * dimension,):
        raise ValueError(
            f"expected {2 * dimension} flattened bounds, got {values.size}"
        )
    lower = values[:dimension]
    upper = values[dimension:]
    if not np.all(np.isfinite(values)):
        raise ValueError("source bounds must be finite")
    if np.any(lower > upper):
        raise ValueError("source lower bound exceeds upper bound")
    return lower, upper


def _flatten_bounds(lower: npt.ArrayLike, upper: npt.ArrayLike) -> list[float]:
    return [
        *(float(value) for value in np.asarray(lower, dtype=float)),
        *(float(value) for value in np.asarray(upper, dtype=float)),
    ]


class CMGDBSuspensionBoxMap:
    """Finite tagged-union image callback for ``CMGDB.AtlasModel``.

    Args:
        system: Hybrid flow, guard, and reset map.
        charts: Base/handle coordinate specification.
        t_star: Positive time of the suspension semiflow map.
        samples_per_axis: Tensor nodes per source coordinate.  At least three
            are required, so every nondegenerate source box has interior
            samples in addition to all corners.
        padding_cells: Target-chart padding measured in dyadic cells at the
            source box's depth.  This is the Atlas analogue of CMGDB's usual
            sample-and-bloat rule.
        diagnostics_limit: Number of ordinary per-source records retained.
            Aggregate counters always include all calls.  When the bridge is
            enabled, every bridge-bearing record is additionally retained in
            ``single_handle_bridge_records`` regardless of this limit, because
            a synthesized carrier without its probes is not auditable.
        single_handle_bridge: Opt in to deterministic bisection of an adjacent
            ``base(2*j)``/``base(2*j+2)`` source-lattice edge.  A successful
            probe adds one tagged full handle to the terminal image only.
        single_handle_bridge_max_bisections: Hard per-edge probe limit for the
            opt-in bridge.  Failure to observe the intermediate handle leaves
            the raw stage gap unresolved.

    Notes:
        Returned hulls are separated by ``(chart, stage)``.  Adjacent tensor
        nodes are inspected for event-stage changes.  A one-stage change adds
        both incident representatives of the appropriate quotient face.  By
        default, a larger jump remains unresolved.  The opt-in bridge can
        discharge exactly a two-stage base-to-base gap using a sampled handle
        witness; all other gaps remain hard failures.
    """

    def __init__(
        self,
        system: HybridSystem,
        charts: SuspensionAtlasCharts,
        t_star: float,
        *,
        samples_per_axis: int | Sequence[int] = 3,
        padding_cells: float = 1.0,
        max_jumps: int | None = None,
        max_step: float | None = None,
        atol: float = 1e-9,
        require_domain_path: bool = False,
        diagnostics_limit: int = 2048,
        single_handle_bridge: bool = False,
        single_handle_bridge_max_bisections: int = 12,
    ) -> None:
        self.system = system
        self.charts = charts
        self.t_star = float(t_star)
        if not np.isfinite(self.t_star) or self.t_star <= 0.0:
            raise ValueError("t_star must be finite and positive")
        if isinstance(samples_per_axis, (int, np.integer)):
            if int(samples_per_axis) < 3:
                raise ValueError("samples_per_axis must be at least three")
            self._samples_per_axis: int | tuple[int, ...] = int(samples_per_axis)
        else:
            counts = tuple(int(value) for value in samples_per_axis)
            if not counts or any(value < 3 for value in counts):
                raise ValueError("every samples_per_axis entry must be at least three")
            self._samples_per_axis = counts
        self.padding_cells = float(padding_cells)
        if not np.isfinite(self.padding_cells) or self.padding_cells < 0.0:
            raise ValueError("padding_cells must be finite and non-negative")
        self.max_jumps = system.max_jumps if max_jumps is None else int(max_jumps)
        if self.max_jumps < 0:
            raise ValueError("max_jumps must be non-negative")
        self.max_step = max_step
        if max_step is not None and (not np.isfinite(max_step) or max_step <= 0.0):
            raise ValueError("max_step must be finite and positive")
        self.atol = float(atol)
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError("atol must be finite and non-negative")
        self.require_domain_path = bool(require_domain_path)
        self.diagnostics_limit = int(diagnostics_limit)
        if self.diagnostics_limit < 0:
            raise ValueError("diagnostics_limit must be non-negative")
        if type(single_handle_bridge) is not bool:
            raise TypeError("single_handle_bridge must be a boolean")
        if (
            isinstance(single_handle_bridge_max_bisections, (bool, np.bool_))
            or not isinstance(single_handle_bridge_max_bisections, (int, np.integer))
            or int(single_handle_bridge_max_bisections) < 1
        ):
            raise ValueError(
                "single_handle_bridge_max_bisections must be a positive integer"
            )
        self.single_handle_bridge = single_handle_bridge
        self.single_handle_bridge_max_bisections = int(
            single_handle_bridge_max_bisections
        )
        self._diagnostics = _MutableDiagnostics()
        self._diagnostics_lock = threading.Lock()

    @property
    def samples_per_axis(self) -> int | tuple[int, ...]:
        return self._samples_per_axis

    def _tensor_shape(self, dimension: int) -> tuple[int, ...]:
        if isinstance(self._samples_per_axis, int):
            return (self._samples_per_axis,) * dimension
        if len(self._samples_per_axis) != dimension:
            raise ValueError(
                "samples_per_axis sequence must match every source chart dimension"
            )
        return self._samples_per_axis

    def diagnostics(self) -> SuspensionBoxMapDiagnostics:
        """Return a thread-safe immutable diagnostics snapshot."""

        with self._diagnostics_lock:
            data = self._diagnostics
            return SuspensionBoxMapDiagnostics(
                single_handle_bridge_enabled=self.single_handle_bridge,
                single_handle_bridge_algorithm_revision=(
                    SINGLE_HANDLE_BRIDGE_ALGORITHM
                ),
                single_handle_bridge_max_bisections=(
                    self.single_handle_bridge_max_bisections
                ),
                single_handle_bridge_assumptions=(
                    SINGLE_HANDLE_BRIDGE_ASSUMPTIONS
                ),
                source_boxes=data.source_boxes,
                sampled_points=data.sampled_points,
                successful_samples=data.successful_samples,
                failed_samples=data.failed_samples,
                returned_pieces=data.returned_pieces,
                empty_images=data.empty_images,
                guard_face_witnesses=data.guard_face_witnesses,
                reset_face_witnesses=data.reset_face_witnesses,
                raw_unresolved_stage_edges=data.raw_unresolved_stage_edges,
                unresolved_stage_edges=data.unresolved_stage_edges,
                single_handle_bridge_attempts=data.single_handle_bridge_attempts,
                synthesized_single_handle_bridges=(
                    data.synthesized_single_handle_bridges
                ),
                single_handle_bridge_probe_points=(
                    data.single_handle_bridge_probe_points
                ),
                single_handle_bridge_failed_probes=(
                    data.single_handle_bridge_failed_probes
                ),
                interior_only_strata=data.interior_only_strata,
                retained_source_records=tuple(data.records),
                single_handle_bridge_records=tuple(data.bridge_records),
            )

    def reset_diagnostics(self) -> None:
        with self._diagnostics_lock:
            self._diagnostics = _MutableDiagnostics()

    def decode_handle(
        self,
        coordinates: npt.ArrayLike,
    ) -> tuple[State, State, float]:
        """Decode handle coordinates as ``(guard, reset, phase)``."""

        guard, phase = self.charts.decode_handle(coordinates)
        reset = np.asarray(self.system.apply_reset_map(guard), dtype=np.float64)
        if reset.shape != guard.shape or not np.all(np.isfinite(reset)):
            raise ValueError("reset_map returned an invalid state")
        return guard, reset, phase

    def evaluate_point(
        self,
        source_chart_id: int,
        coordinates: npt.ArrayLike,
    ) -> SuspensionSample:
        """Evaluate one exact/reference suspension point used by the sampler.

        This public method is useful for independent endpoint audits.  It does
        not update box-map diagnostics and it does not claim rigorous ODE
        enclosure; it uses the same event solver as the tensor callback.
        """

        chart_id = int(source_chart_id)
        dimension = len(self.charts.bounds_for(chart_id))
        point = np.asarray(coordinates, dtype=np.float64)
        if point.shape != (dimension,) or not np.all(np.isfinite(point)):
            raise ValueError("source coordinates have the wrong dimension or are non-finite")
        sample, _faces = self._evaluate_point(chart_id, point)
        return sample

    def encode_endpoint(
        self,
        sample: SuspensionSample,
    ) -> tuple[int, tuple[float, ...], SuspensionStratum]:
        """Encode a sampled endpoint with its Atlas tag and event stage."""

        observation = self._observation(
            grid_index=(),
            source_point=np.asarray([], dtype=np.float64),
            corner=False,
            sample=sample,
        )
        return (
            observation.stratum.chart_id,
            observation.target_coordinates,
            observation.stratum,
        )

    def __call__(
        self,
        source_chart_id: int,
        source_bounds: Sequence[float],
    ) -> list[TaggedRectangle]:
        chart_id = int(source_chart_id)
        chart_bounds = self.charts.bounds_for(chart_id)
        dimension = len(chart_bounds)
        lower, upper = _split_bounds(source_bounds, dimension)
        tensor_shape = self._tensor_shape(dimension)
        axes = tuple(
            np.linspace(lower[axis], upper[axis], tensor_shape[axis])
            for axis in range(dimension)
        )

        observations: dict[tuple[int, ...], _EndpointObservation] = {}
        failures: list[SamplingFailure] = []
        explicit_faces: list[_FaceWitness] = []
        corner_strata: set[SuspensionStratum] = set()

        for grid_index in itertools.product(
            *(range(count) for count in tensor_shape)
        ):
            point = np.asarray(
                [axes[axis][grid_index[axis]] for axis in range(dimension)],
                dtype=np.float64,
            )
            corner = all(
                index in (0, tensor_shape[axis] - 1)
                for axis, index in enumerate(grid_index)
            )
            try:
                sample, point_faces = self._evaluate_point(chart_id, point)
                observation = self._observation(
                    grid_index=tuple(int(value) for value in grid_index),
                    source_point=point,
                    corner=corner,
                    sample=sample,
                )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                failures.append(
                    SamplingFailure(
                        source_point=tuple(float(value) for value in point),
                        reason=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            observations[observation.grid_index] = observation
            explicit_faces.extend(
                replace(
                    witness,
                    incident_indices=frozenset((observation.grid_index,)),
                )
                for witness in point_faces
            )
            if corner:
                corner_strata.add(observation.stratum)

        grouped: dict[SuspensionStratum, list[_EndpointObservation]] = defaultdict(list)
        for observation in observations.values():
            grouped[observation.stratum].append(observation)

        (
            face_witnesses,
            raw_unresolved,
            unresolved,
            bridge_attempts,
        ) = self._adjacent_face_witnesses(
            chart_id,
            observations,
            tensor_shape,
        )
        face_witnesses.extend(explicit_faces)
        face_witnesses = self._deduplicate_faces(face_witnesses)

        # A seam point appended as a separate rectangle is not enough: the
        # sampled branch rectangle must itself reach that seam.  Attach each
        # incident face coordinate to the source-lattice component that
        # produced it before taking target hulls.  Standalone copies are still
        # returned below so both quotient representatives are always present.
        component_face_coordinates: dict[
            tuple[int, ...], list[tuple[float, ...]]
        ] = defaultdict(list)
        for witness in face_witnesses:
            for incident_index in witness.incident_indices:
                observation = observations.get(incident_index)
                if observation is None:
                    continue
                component_face_coordinates[incident_index].append(
                    self._face_coordinates_for_stratum(
                        witness,
                        observation.stratum,
                    )
                )

        source_fraction = self._source_depth_fraction(
            chart_id,
            lower,
            upper,
        )
        pieces: list[TaggedRectangle] = []
        for stratum, component in self._stratum_components(observations):
            component_coordinates = [
                observation.target_coordinates for observation in component
            ]
            component_coordinates.extend(
                coordinate
                for observation in component
                for coordinate in component_face_coordinates.get(
                    observation.grid_index,
                    (),
                )
            )
            coordinates = np.asarray(component_coordinates, dtype=np.float64)
            pieces.append(
                self._padded_piece(
                    stratum.chart_id,
                    np.min(coordinates, axis=0),
                    np.max(coordinates, axis=0),
                    source_fraction,
                )
            )

        emitted_bridge_attempts: list[SingleHandleBridgeDiagnostic] = []
        for bridge in bridge_attempts:
            if not bridge.synthesized:
                emitted_bridge_attempts.append(bridge)
                continue
            raw_bounds = bridge.raw_handle_carrier_bounds
            if raw_bounds is None:  # pragma: no cover - internal invariant
                raise AssertionError("a synthesized bridge lacks its handle carrier")
            carrier_lower, carrier_upper = _split_bounds(
                raw_bounds,
                self.charts.handle_dimension,
            )
            carrier_piece = self._padded_piece(
                self.charts.handle_chart_id,
                carrier_lower,
                carrier_upper,
                source_fraction,
            )
            pieces.append(carrier_piece)
            emitted_bridge_attempts.append(
                replace(
                    bridge,
                    emitted_handle_carrier_bounds=tuple(carrier_piece[1]),
                )
            )

        guard_faces = 0
        reset_faces = 0
        for witness in face_witnesses:
            if witness.face == "guard":
                guard_faces += 1
                pieces.extend(self._guard_face_pieces(witness, source_fraction))
            elif witness.face == "reset":
                reset_faces += 1
                pieces.extend(self._reset_face_pieces(witness, source_fraction))
            else:  # pragma: no cover - internal invariant
                raise AssertionError(f"unknown quotient face {witness.face!r}")
        pieces = self._deduplicate_pieces(pieces)

        all_strata = tuple(sorted(grouped))
        interior_only = tuple(
            stratum for stratum in all_strata if stratum not in corner_strata
        )
        record = SourceBoxMapDiagnostic(
            source_chart_id=chart_id,
            source_bounds=tuple(float(value) for value in source_bounds),
            tensor_shape=tensor_shape,
            sample_count=int(np.prod(tensor_shape)),
            successful_samples=len(observations),
            corner_samples=sum(observation.corner for observation in observations.values()),
            strata=all_strata,
            interior_only_strata=interior_only,
            guard_face_witnesses=guard_faces,
            reset_face_witnesses=reset_faces,
            raw_unresolved_stage_edges=tuple(sorted(raw_unresolved)),
            unresolved_stage_edges=tuple(sorted(unresolved)),
            single_handle_bridge_attempts=tuple(emitted_bridge_attempts),
            failures=tuple(failures),
            returned_pieces=len(pieces),
            empty_image=not pieces,
        )
        self._record(record)
        return pieces

    @staticmethod
    def _stratum_components(
        observations: Mapping[tuple[int, ...], _EndpointObservation],
    ) -> list[tuple[SuspensionStratum, list[_EndpointObservation]]]:
        """Return lattice-connected components within each observed stratum.

        Even one event itinerary can occupy disjoint regions of a source box.
        Hulling those regions together would be an avoidable overestimate, so
        each nearest-neighbor tensor component gets its own target rectangle.
        """

        remaining = set(observations)
        components: list[tuple[SuspensionStratum, list[_EndpointObservation]]] = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            stratum = observations[seed].stratum
            frontier = [seed]
            indices: list[tuple[int, ...]] = []
            while frontier:
                current = frontier.pop()
                indices.append(current)
                for axis in range(len(current)):
                    for offset in (-1, 1):
                        neighbor = list(current)
                        neighbor[axis] += offset
                        neighbor_key = tuple(neighbor)
                        if neighbor_key not in remaining:
                            continue
                        if observations[neighbor_key].stratum != stratum:
                            continue
                        remaining.remove(neighbor_key)
                        frontier.append(neighbor_key)
            components.append(
                (stratum, [observations[index] for index in sorted(indices)])
            )
        components.sort(
            key=lambda value: (
                value[0],
                min(observation.grid_index for observation in value[1]),
            )
        )
        return components

    def _evaluate_point(
        self,
        source_chart_id: int,
        point: State,
    ) -> tuple[SuspensionSample, list[_FaceWitness]]:
        if source_chart_id == self.charts.base_chart_id:
            sample = simulate_suspension_endpoint(
                self.system,
                point,
                self.t_star,
                max_jumps=self.max_jumps,
                max_step=self.max_step,
                atol=self.atol,
                require_domain_path=self.require_domain_path,
            )
            faces: list[_FaceWitness] = []
            if isinstance(sample, BaseSuspensionSample):
                # A base endpoint also lies on the guard face of the handle
                # only if it is a guard point: on the event surface with the
                # flow not moving against the event direction.
                try:
                    guard_face = bool(
                        self.system.on_guard(
                            sample.continuous_time,
                            sample.state,
                            tolerance=self.atol,
                        )
                    )
                except (TypeError, ValueError, FloatingPointError):
                    guard_face = False
                if guard_face:
                    reset = np.asarray(
                        self.system.apply_reset_map(sample.state),
                        dtype=np.float64,
                    )
                    self._validated_chart_point(
                        self.charts.base_chart_id,
                        sample.state,
                        label="guard face",
                    )
                    self._validated_chart_point(
                        self.charts.base_chart_id,
                        reset,
                        label="reset face",
                    )
                    self._validated_chart_point(
                        self.charts.handle_chart_id,
                        self.charts.encode_handle(sample.state, 0.0),
                        label="handle guard face",
                    )
                    faces.append(
                        _FaceWitness(
                            face="guard",
                            guard_state=tuple(float(value) for value in sample.state),
                            reset_state=tuple(float(value) for value in reset),
                            jump_index=sample.jumps_completed,
                        )
                    )
            return sample, faces

        if source_chart_id != self.charts.handle_chart_id:
            raise ValueError(f"unknown suspension chart id {source_chart_id}")
        raw_phase = float(point[-1])
        if raw_phase < -self.atol or raw_phase > 1.0 + self.atol:
            raise ValueError("handle source phase lies outside [0, 1]")
        clipped = np.asarray(point, dtype=np.float64).copy()
        clipped[-1] = min(max(raw_phase, 0.0), 1.0)
        guard, reset, phase = self.decode_handle(clipped)
        return self._endpoint_from_handle(guard, reset, phase)

    def _endpoint_from_handle(
        self,
        guard: State,
        reset: State,
        phase: float,
    ) -> tuple[SuspensionSample, list[_FaceWitness]]:
        target_phase = phase + self.t_star
        tolerance = self.atol * max(1.0, abs(target_phase))
        if target_phase < 1.0 - tolerance:
            return (
                HandleSuspensionSample(
                    guard_state=guard,
                    reset_state=reset,
                    phase=float(target_phase),
                    total_time=self.t_star,
                    continuous_time=0.0,
                    jump_index=0,
                ),
                [],
            )

        remaining = max(0.0, target_phase - 1.0)
        if remaining <= tolerance:
            return (
                BaseSuspensionSample(
                    state=reset,
                    total_time=self.t_star,
                    continuous_time=0.0,
                    jumps_completed=1,
                ),
                [
                    _FaceWitness(
                        face="reset",
                        guard_state=tuple(float(value) for value in guard),
                        reset_state=tuple(float(value) for value in reset),
                        jump_index=0,
                    )
                ],
            )

        sample = simulate_suspension_endpoint(
            self.system,
            reset,
            remaining,
            max_jumps=self.max_jumps,
            max_step=self.max_step,
            atol=self.atol,
            require_domain_path=self.require_domain_path,
        )
        if isinstance(sample, BaseSuspensionSample):
            shifted: SuspensionSample = BaseSuspensionSample(
                state=sample.state,
                total_time=self.t_star,
                continuous_time=sample.continuous_time,
                jumps_completed=sample.jumps_completed + 1,
            )
        else:
            shifted = HandleSuspensionSample(
                guard_state=sample.guard_state,
                reset_state=sample.reset_state,
                phase=sample.phase,
                total_time=self.t_star,
                continuous_time=sample.continuous_time,
                jump_index=sample.jump_index + 1,
            )
        return shifted, []

    def _observation(
        self,
        *,
        grid_index: tuple[int, ...],
        source_point: State,
        corner: bool,
        sample: SuspensionSample,
    ) -> _EndpointObservation:
        if isinstance(sample, BaseSuspensionSample):
            coordinates = np.asarray(sample.state, dtype=np.float64)
            if coordinates.shape != (self.charts.base_dimension,):
                raise ValueError("base endpoint has the wrong dimension")
            if not np.all(np.isfinite(coordinates)):
                raise ValueError("base endpoint is non-finite")
            coordinates = self._validated_domain_state(
                coordinates,
                label="base endpoint",
            )
            coordinates = self._validated_chart_point(
                self.charts.base_chart_id,
                coordinates,
                label="base endpoint",
            )
            stratum = SuspensionStratum(
                chart_id=self.charts.base_chart_id,
                stage=2 * int(sample.jumps_completed),
                kind="base",
            )
        elif isinstance(sample, HandleSuspensionSample):
            if not 0.0 < sample.phase < 1.0:
                raise ValueError("handle endpoint phase must lie strictly inside (0, 1)")
            intrinsic = np.asarray(
                self.charts.guard_coordinates(sample.guard_state),
                dtype=np.float64,
            )
            if intrinsic.shape != (len(self.charts.guard_bounds),):
                raise ValueError("guard_coordinates returned the wrong dimension")
            coordinates = np.concatenate((intrinsic, [sample.phase]))
            if not np.all(np.isfinite(coordinates)):
                raise ValueError("handle endpoint is non-finite")
            guard = np.asarray(sample.guard_state, dtype=np.float64)
            reset = np.asarray(sample.reset_state, dtype=np.float64)
            if guard.shape != (self.charts.base_dimension,) or reset.shape != guard.shape:
                raise ValueError("handle endpoint guard/reset states have the wrong dimension")
            if not np.all(np.isfinite(guard)) or not np.all(np.isfinite(reset)):
                raise ValueError("handle endpoint guard/reset state is non-finite")
            guard = self._validated_domain_state(
                guard,
                label="handle guard face",
            )
            reset = self._validated_domain_state(
                reset,
                label="handle reset face",
            )
            self._validated_chart_point(
                self.charts.base_chart_id,
                guard,
                label="handle guard face",
            )
            self._validated_chart_point(
                self.charts.base_chart_id,
                reset,
                label="handle reset face",
            )
            coordinates = self._validated_chart_point(
                self.charts.handle_chart_id,
                coordinates,
                label="handle endpoint",
            )
            stratum = SuspensionStratum(
                chart_id=self.charts.handle_chart_id,
                stage=2 * int(sample.jump_index) + 1,
                kind="handle",
            )
        else:  # pragma: no cover - closed union from sampled_suspension
            raise TypeError(f"unsupported suspension endpoint {type(sample)!r}")
        return _EndpointObservation(
            grid_index=grid_index,
            source_point=tuple(float(value) for value in source_point),
            corner=corner,
            sample=sample,
            stratum=stratum,
            target_coordinates=tuple(float(value) for value in coordinates),
        )

    def _validated_domain_state(
        self,
        state: npt.ArrayLike,
        *,
        label: str,
    ) -> State:
        """Validate a state against system bounds with roundoff tolerance."""

        point = np.asarray(state, dtype=np.float64)
        if point.shape != (self.charts.base_dimension,) or not np.all(
            np.isfinite(point)
        ):
            raise ValueError(f"{label} has the wrong dimension or is non-finite")
        if self.system.domain_bounds is None:
            return point
        bounds = np.asarray(self.system.domain_bounds, dtype=np.float64)
        if bounds.shape != (self.charts.base_dimension, 2):
            raise ValueError("system domain bounds do not match the base dimension")
        scale = np.maximum.reduce(
            (
                np.ones(len(bounds), dtype=np.float64),
                np.abs(bounds[:, 0]),
                np.abs(bounds[:, 1]),
                np.abs(point),
            )
        )
        tolerance = self.atol * scale
        if np.any(point < bounds[:, 0] - tolerance) or np.any(
            point > bounds[:, 1] + tolerance
        ):
            raise ValueError(f"{label} exits the declared state space")
        clipped = np.minimum(np.maximum(point, bounds[:, 0]), bounds[:, 1])
        predicate = getattr(self.system, "domain_predicate", None)
        if predicate is not None and not bool(predicate(clipped)):
            raise ValueError(f"{label} exits the declared nonrectangular state space")
        return clipped

    def _validated_chart_point(
        self,
        chart_id: int,
        coordinates: npt.ArrayLike,
        *,
        label: str,
    ) -> State:
        """Validate an unpadded point against its Atlas chart and clip roundoff."""

        point = np.asarray(coordinates, dtype=np.float64)
        bounds = np.asarray(self.charts.bounds_for(chart_id), dtype=np.float64)
        if point.shape != (len(bounds),) or not np.all(np.isfinite(point)):
            raise ValueError(f"{label} has the wrong dimension or is non-finite")
        scale = np.maximum.reduce(
            (
                np.ones(len(bounds), dtype=np.float64),
                np.abs(bounds[:, 0]),
                np.abs(bounds[:, 1]),
                np.abs(point),
            )
        )
        tolerance = self.atol * scale
        if np.any(point < bounds[:, 0] - tolerance) or np.any(
            point > bounds[:, 1] + tolerance
        ):
            raise ValueError(
                f"{label} lies outside target Atlas chart {chart_id}"
            )
        return np.minimum(np.maximum(point, bounds[:, 0]), bounds[:, 1])

    def _adjacent_face_witnesses(
        self,
        source_chart_id: int,
        observations: Mapping[tuple[int, ...], _EndpointObservation],
        tensor_shape: tuple[int, ...],
    ) -> tuple[
        list[_FaceWitness],
        set[tuple[int, int]],
        set[tuple[int, int]],
        list[SingleHandleBridgeDiagnostic],
    ]:
        witnesses: list[_FaceWitness] = []
        raw_unresolved: set[tuple[int, int]] = set()
        residual_unresolved: set[tuple[int, int]] = set()
        bridge_attempts: list[SingleHandleBridgeDiagnostic] = []
        for grid_index, first in observations.items():
            for axis in range(len(tensor_shape)):
                if grid_index[axis] + 1 >= tensor_shape[axis]:
                    continue
                neighbor_index = list(grid_index)
                neighbor_index[axis] += 1
                second = observations.get(tuple(neighbor_index))
                if second is None:
                    continue
                first_stage = first.stratum.stage
                second_stage = second.stratum.stage
                difference = abs(first_stage - second_stage)
                if difference == 0:
                    continue
                if difference > 1:
                    stage_edge = tuple(sorted((first_stage, second_stage)))
                    raw_unresolved.add(stage_edge)
                    if self.single_handle_bridge and difference == 2:
                        bridge = self._attempt_single_handle_bridge(
                            source_chart_id,
                            first,
                            second,
                        )
                        bridge_attempts.append(bridge)
                        if bridge.synthesized:
                            if bridge.guard_state is None or bridge.reset_state is None:
                                raise AssertionError(
                                    "a synthesized bridge lacks guard/reset provenance"
                                )
                            incident = bridge.incident_grid_indices
                            witnesses.extend(
                                (
                                    _FaceWitness(
                                        face="guard",
                                        guard_state=bridge.guard_state,
                                        reset_state=bridge.reset_state,
                                        jump_index=bridge.lower_stage // 2,
                                        incident_indices=frozenset((incident[0],)),
                                    ),
                                    _FaceWitness(
                                        face="reset",
                                        guard_state=bridge.guard_state,
                                        reset_state=bridge.reset_state,
                                        jump_index=bridge.lower_stage // 2,
                                        incident_indices=frozenset((incident[1],)),
                                    ),
                                )
                            )
                            continue
                    # Gaps larger than two, non-base gaps, failed probes, and
                    # disabled bridging all remain hard acceptance failures.
                    residual_unresolved.add(stage_edge)
                    continue
                lower, upper = sorted(
                    (first, second),
                    key=lambda observation: observation.stratum.stage,
                )
                handle = (
                    lower.sample
                    if isinstance(lower.sample, HandleSuspensionSample)
                    else upper.sample
                )
                if not isinstance(handle, HandleSuspensionSample):
                    # Consecutive stages always contain exactly one handle.
                    stage_edge = (lower.stratum.stage, upper.stratum.stage)
                    raw_unresolved.add(stage_edge)
                    residual_unresolved.add(stage_edge)
                    continue
                face = "guard" if lower.stratum.kind == "base" else "reset"
                witnesses.append(
                    _FaceWitness(
                        face=face,
                        guard_state=tuple(float(value) for value in handle.guard_state),
                        reset_state=tuple(float(value) for value in handle.reset_state),
                        jump_index=int(handle.jump_index),
                        incident_indices=frozenset(
                            (first.grid_index, second.grid_index)
                        ),
                    )
                )
        return witnesses, raw_unresolved, residual_unresolved, bridge_attempts

    def _attempt_single_handle_bridge(
        self,
        source_chart_id: int,
        first: _EndpointObservation,
        second: _EndpointObservation,
    ) -> SingleHandleBridgeDiagnostic:
        """Bisect one source edge until its skipped terminal handle is sampled.

        This routine only accepts ``base(j) -> base(j+1)`` (stage ``2j`` to
        ``2j+2``).  A successful witness supplies one tagged handle carrier
        and its two quotient faces to the *terminal value* of the source box.
        It does not create a time-ordered path or any intermediate graph edge.
        """

        lower, upper = sorted(
            (first, second),
            key=lambda observation: observation.stratum.stage,
        )
        lower_stage = lower.stratum.stage
        upper_stage = upper.stratum.stage
        skipped_stage = lower_stage + 1
        common = {
            "algorithm_revision": SINGLE_HANDLE_BRIDGE_ALGORITHM,
            "incident_grid_indices": (lower.grid_index, upper.grid_index),
            "incident_source_points": (lower.source_point, upper.source_point),
            "lower_stage": lower_stage,
            "skipped_handle_stage": skipped_stage,
            "upper_stage": upper_stage,
            "max_bisections": self.single_handle_bridge_max_bisections,
            "witness_source_point": None,
            "guard_state": None,
            "reset_state": None,
            "witness_phase": None,
            "raw_handle_carrier_bounds": None,
            "emitted_handle_carrier_bounds": None,
        }
        if (
            upper_stage - lower_stage != 2
            or lower_stage % 2 != 0
            or lower.stratum.kind != "base"
            or upper.stratum.kind != "base"
            or lower.stratum.chart_id != self.charts.base_chart_id
            or upper.stratum.chart_id != self.charts.base_chart_id
        ):
            return SingleHandleBridgeDiagnostic(
                status="rejected_not_adjacent_base_branches",
                probes=(),
                **common,
            )

        lower_point = np.asarray(lower.source_point, dtype=np.float64)
        upper_point = np.asarray(upper.source_point, dtype=np.float64)
        probes: list[SingleHandleBridgeProbe] = []
        for depth in range(1, self.single_handle_bridge_max_bisections + 1):
            midpoint = 0.5 * (lower_point + upper_point)
            source_point = tuple(float(value) for value in midpoint)
            try:
                sample, _explicit_faces = self._evaluate_point(
                    source_chart_id,
                    midpoint,
                )
                observation = self._observation(
                    grid_index=(),
                    source_point=midpoint,
                    corner=False,
                    sample=sample,
                )
            except (RuntimeError, ValueError, FloatingPointError) as error:
                probes.append(
                    SingleHandleBridgeProbe(
                        bisection_depth=depth,
                        source_point=source_point,
                        stage=None,
                        kind=None,
                        failure=f"{type(error).__name__}: {error}",
                    )
                )
                return SingleHandleBridgeDiagnostic(
                    status="probe_failed",
                    probes=tuple(probes),
                    **common,
                )

            probes.append(
                SingleHandleBridgeProbe(
                    bisection_depth=depth,
                    source_point=source_point,
                    stage=observation.stratum.stage,
                    kind=observation.stratum.kind,
                    failure=None,
                )
            )
            if observation.stratum.stage == skipped_stage:
                if not isinstance(sample, HandleSuspensionSample):
                    return SingleHandleBridgeDiagnostic(
                        status="rejected_intermediate_not_handle",
                        probes=tuple(probes),
                        **common,
                    )
                intrinsic = np.asarray(
                    self.charts.guard_coordinates(sample.guard_state),
                    dtype=np.float64,
                )
                carrier_lower = self._validated_chart_point(
                    self.charts.handle_chart_id,
                    np.concatenate((intrinsic, [0.0])),
                    label="single-handle bridge lower face",
                )
                carrier_upper = self._validated_chart_point(
                    self.charts.handle_chart_id,
                    np.concatenate((intrinsic, [1.0])),
                    label="single-handle bridge upper face",
                )
                return SingleHandleBridgeDiagnostic(
                    status="synthesized",
                    probes=tuple(probes),
                    witness_source_point=source_point,
                    guard_state=tuple(
                        float(value) for value in sample.guard_state
                    ),
                    reset_state=tuple(
                        float(value) for value in sample.reset_state
                    ),
                    witness_phase=float(sample.phase),
                    raw_handle_carrier_bounds=tuple(
                        _flatten_bounds(carrier_lower, carrier_upper)
                    ),
                    emitted_handle_carrier_bounds=None,
                    **{
                        key: value
                        for key, value in common.items()
                        if key
                        not in {
                            "witness_source_point",
                            "guard_state",
                            "reset_state",
                            "witness_phase",
                            "raw_handle_carrier_bounds",
                            "emitted_handle_carrier_bounds",
                        }
                    },
                )
            if (
                observation.stratum.stage == lower_stage
                and observation.stratum.kind == "base"
            ):
                lower_point = midpoint
                continue
            if (
                observation.stratum.stage == upper_stage
                and observation.stratum.kind == "base"
            ):
                upper_point = midpoint
                continue
            return SingleHandleBridgeDiagnostic(
                status="rejected_unexpected_probe_stage",
                probes=tuple(probes),
                **common,
            )

        return SingleHandleBridgeDiagnostic(
            status="intermediate_handle_not_found",
            probes=tuple(probes),
            **common,
        )

    @staticmethod
    def _deduplicate_faces(witnesses: Sequence[_FaceWitness]) -> list[_FaceWitness]:
        unique: dict[tuple[object, ...], _FaceWitness] = {}
        for witness in witnesses:
            key = (
                witness.face,
                witness.jump_index,
                *(round(value, 14) for value in witness.guard_state),
                *(round(value, 14) for value in witness.reset_state),
            )
            previous = unique.get(key)
            if previous is None:
                unique[key] = witness
            else:
                unique[key] = replace(
                    previous,
                    incident_indices=(
                        previous.incident_indices | witness.incident_indices
                    ),
                )
        return list(unique.values())

    def _face_coordinates_for_stratum(
        self,
        witness: _FaceWitness,
        stratum: SuspensionStratum,
    ) -> tuple[float, ...]:
        """Return the representative of ``witness`` in one incident chart."""

        guard = np.asarray(witness.guard_state, dtype=np.float64)
        if stratum.kind == "base":
            state = guard if witness.face == "guard" else np.asarray(
                witness.reset_state,
                dtype=np.float64,
            )
            state = self._validated_chart_point(
                self.charts.base_chart_id,
                state,
                label=f"{witness.face} face",
            )
            return tuple(float(value) for value in state)
        phase = 0.0 if witness.face == "guard" else 1.0
        handle = self._validated_chart_point(
            self.charts.handle_chart_id,
            self.charts.encode_handle(guard, phase),
            label=f"handle {witness.face} face",
        )
        return tuple(float(value) for value in handle)

    def _source_depth_fraction(
        self,
        chart_id: int,
        lower: State,
        upper: State,
    ) -> float:
        full = np.asarray(self.charts.bounds_for(chart_id), dtype=np.float64)
        fractions = (upper - lower) / (full[:, 1] - full[:, 0])
        return float(np.max(fractions))

    def _padded_piece(
        self,
        chart_id: int,
        lower: npt.ArrayLike,
        upper: npt.ArrayLike,
        source_fraction: float,
    ) -> TaggedRectangle:
        target_bounds = np.asarray(self.charts.bounds_for(chart_id), dtype=np.float64)
        padding = (
            self.padding_cells
            * source_fraction
            * (target_bounds[:, 1] - target_bounds[:, 0])
        )
        return (
            chart_id,
            _flatten_bounds(
                np.asarray(lower, dtype=np.float64) - padding,
                np.asarray(upper, dtype=np.float64) + padding,
            ),
        )

    def _guard_face_pieces(
        self,
        witness: _FaceWitness,
        source_fraction: float,
    ) -> list[TaggedRectangle]:
        guard = self._validated_chart_point(
            self.charts.base_chart_id,
            witness.guard_state,
            label="guard face",
        )
        handle = self._validated_chart_point(
            self.charts.handle_chart_id,
            self.charts.encode_handle(guard, 0.0),
            label="handle guard face",
        )
        return [
            self._padded_piece(
                self.charts.base_chart_id,
                guard,
                guard,
                source_fraction,
            ),
            self._padded_piece(
                self.charts.handle_chart_id,
                handle,
                handle,
                source_fraction,
            ),
        ]

    def _reset_face_pieces(
        self,
        witness: _FaceWitness,
        source_fraction: float,
    ) -> list[TaggedRectangle]:
        guard = self._validated_chart_point(
            self.charts.base_chart_id,
            witness.guard_state,
            label="guard face",
        )
        reset = self._validated_chart_point(
            self.charts.base_chart_id,
            witness.reset_state,
            label="reset face",
        )
        handle = self._validated_chart_point(
            self.charts.handle_chart_id,
            self.charts.encode_handle(guard, 1.0),
            label="handle reset face",
        )
        return [
            self._padded_piece(
                self.charts.handle_chart_id,
                handle,
                handle,
                source_fraction,
            ),
            self._padded_piece(
                self.charts.base_chart_id,
                reset,
                reset,
                source_fraction,
            ),
        ]

    @staticmethod
    def _deduplicate_pieces(
        pieces: Sequence[TaggedRectangle],
    ) -> list[TaggedRectangle]:
        unique: dict[tuple[Hashable, ...], TaggedRectangle] = {}
        for chart_id, bounds in pieces:
            key = (chart_id, *(round(float(value), 14) for value in bounds))
            unique[key] = (int(chart_id), [float(value) for value in bounds])
        return list(unique.values())

    def _record(self, record: SourceBoxMapDiagnostic) -> None:
        with self._diagnostics_lock:
            data = self._diagnostics
            data.source_boxes += 1
            data.sampled_points += record.sample_count
            data.successful_samples += record.successful_samples
            data.failed_samples += len(record.failures)
            data.returned_pieces += record.returned_pieces
            data.empty_images += int(record.empty_image)
            data.guard_face_witnesses += record.guard_face_witnesses
            data.reset_face_witnesses += record.reset_face_witnesses
            data.raw_unresolved_stage_edges += len(
                record.raw_unresolved_stage_edges
            )
            data.unresolved_stage_edges += len(record.unresolved_stage_edges)
            data.single_handle_bridge_attempts += len(
                record.single_handle_bridge_attempts
            )
            data.synthesized_single_handle_bridges += sum(
                bridge.synthesized
                for bridge in record.single_handle_bridge_attempts
            )
            data.single_handle_bridge_probe_points += sum(
                len(bridge.probes)
                for bridge in record.single_handle_bridge_attempts
            )
            data.single_handle_bridge_failed_probes += sum(
                bridge.failed_probe_count
                for bridge in record.single_handle_bridge_attempts
            )
            data.interior_only_strata += len(record.interior_only_strata)
            if len(data.records) < self.diagnostics_limit:
                data.records.append(record)
            if record.single_handle_bridge_attempts:
                # Terminal bridges are an explicit sampled-cover assumption,
                # so their full source/probe provenance must not disappear
                # merely because the ordinary diagnostics window filled up.
                data.bridge_records.append(record)


def build_cmgdb_atlas_model(
    box_map: CMGDBSuspensionBoxMap,
    *,
    depth: int | None = None,
    depth_min: int | None = None,
    depth_max: int | None = None,
    depth_init: int | None = None,
    subdivision_limit: int = 10000,
    active_dyadic_cells: Sequence[tuple[int, int, Sequence[int]]] | None = None,
) -> Any:
    """Construct a native ``CMGDB.AtlasModel`` around ``box_map``.

    Give either one fixed ``depth`` or all three adaptive depths.  When
    ``active_dyadic_cells`` is supplied, CMGDB constructs that selected tagged
    dyadic family directly before installing the map; charts omitted from the
    family are empty.  CMGDB is imported lazily so the adapter and its tests
    remain usable before the local CMGDB extension has been built.
    """

    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "CMGDB is not installed; build the local fork before constructing "
            "an AtlasModel"
        ) from error

    adaptive = (depth_min, depth_max, depth_init)
    if depth is not None:
        if any(value is not None for value in adaptive):
            raise ValueError("give fixed depth or adaptive depths, not both")
        model = CMGDB.AtlasModel(int(depth))
    else:
        if any(value is None for value in adaptive):
            raise ValueError(
                "adaptive construction requires depth_min, depth_max, and depth_init"
            )
        model = CMGDB.AtlasModel(
            int(depth_min),
            int(depth_max),
            int(depth_init),
            int(subdivision_limit),
        )

    charts = box_map.charts
    for chart_id in (charts.base_chart_id, charts.handle_chart_id):
        bounds = charts.bounds_for(chart_id)
        lower = [float(interval[0]) for interval in bounds]
        upper = [float(interval[1]) for interval in bounds]
        periodic = list(charts.periodic_for(chart_id))
        model.add_chart(chart_id, lower, upper, periodic)
    if active_dyadic_cells is not None:
        if not hasattr(model, "set_active_subgrid"):
            raise RuntimeError(
                "the installed CMGDB AtlasModel lacks set_active_subgrid"
            )
        model.set_active_subgrid(
            [
                (int(chart_id), int(axis_depth), [int(value) for value in indices])
                for chart_id, axis_depth, indices in active_dyadic_cells
            ]
        )
    model.set_map(box_map)
    return model


__all__ = [
    "CMGDBSuspensionBoxMap",
    "SINGLE_HANDLE_BRIDGE_ALGORITHM",
    "SINGLE_HANDLE_BRIDGE_ASSUMPTIONS",
    "SamplingFailure",
    "SingleHandleBridgeDiagnostic",
    "SingleHandleBridgeProbe",
    "SourceBoxMapDiagnostic",
    "SuspensionAtlasCharts",
    "SuspensionBoxMapDiagnostics",
    "SuspensionStratum",
    "TaggedRectangle",
    "build_cmgdb_atlas_model",
]
