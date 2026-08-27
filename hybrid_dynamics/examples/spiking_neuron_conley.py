"""Finite-relation Conley audit for the persisted spiking-neuron Atlas.

The input is the authenticated, complete adaptive ``MapGraph`` artifact.  The
standard top-cell pair is formed without pruning,

``S=M, X=S union F(S), A=F(S) minus S``.

The actual base and handle rectangles in ``X`` are then used to construct the
reset-quotient nerve.  In particular, the interior reset seam is accepted
only through the constructor-verified two-sided subcomplex opt-in.  This
module computes the shift class of the resulting finite relation over GF(5);
it deliberately makes no rigorous ODE-enclosure or continuous-system Conley
claim and never attaches an analytic label.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..src.suspension_complex import CMGDBRelativeHomologyPayload
from .physical_conley import (
    AtlasNerveFiniteRelationAudit,
    AtlasRelationSnapshot,
    AtlasTopCellRecord,
    audit_atlas_nerve_finite_relation,
)
from .spiking_neuron_atlas import (
    BASE_CHART_ID,
    HANDLE_CHART_ID,
    SINGLE_HANDLE_BRIDGE_ALGORITHM,
    AtlasNeuronCell,
    _cells_containing_point,
    compute_neuron_reference_cycle,
    spiking_neuron_atlas_charts,
    spiking_neuron_atlas_reset_gluing,
    validate_spiking_neuron_provenance,
)


CONLEY_PROTOCOL_REVISION = "spiking-neuron-finite-relation-conley-v2"
CONLEY_CHECKPOINT_SCHEMA = "spiking-neuron-conley-chain-checkpoint-v2"
REFERENCE_BASE_SAMPLES = 129
REFERENCE_HANDLE_SAMPLES = 65


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SpikingNeuronConleyInput:
    """Strictly loaded relation snapshot and its authenticated bindings."""

    stage_directory: Path
    snapshot: AtlasRelationSnapshot
    stage_summary: Mapping[str, object]
    relation_reference: Mapping[str, object]
    provenance_reference: Mapping[str, object]


def load_spiking_neuron_conley_input(
    stage_directory: str | Path,
) -> SpikingNeuronConleyInput:
    """Load and cross-check the stage JSON, CSR, and per-source provenance."""

    stage = Path(stage_directory)
    summary = json.loads((stage / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema") != "spiking-neuron-atlas-acceptance-v1":
        raise ValueError("unsupported spiking-neuron acceptance summary")
    if not summary.get("stage_gates_passed"):
        raise ValueError("finite-relation Conley audit requires a gate-passing stage")
    if not summary.get("single_handle_bridge_enabled"):
        raise ValueError("the reviewed neuron stage must declare its terminal carrier")
    if summary.get("unresolved_sources") != 0:
        raise ValueError("the reviewed neuron stage still has unresolved sources")
    if summary.get("failed_single_handle_bridge_sources") != 0:
        raise ValueError("the reviewed neuron stage has failed terminal carriers")
    if summary.get("analytic_conley_label_attached") is not False:
        raise ValueError("an analytic Conley label was unexpectedly attached")

    relation_reference = summary.get("relation_csr")
    if not isinstance(relation_reference, Mapping):
        raise ValueError("acceptance summary lacks its CSR relation reference")
    relation_fingerprint = str(relation_reference["fingerprint"])
    relation_path = stage / str(relation_reference["path"])
    metadata = json.loads(
        (relation_path / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("schema") != "cmgdb-mapgraph-csr-checkpoint-v1":
        raise ValueError("unsupported CMGDB CSR checkpoint")
    if metadata["fingerprint"]["sha256"] != relation_fingerprint:
        raise ValueError("summary and CSR relation fingerprints disagree")
    configuration = metadata["configuration"]
    if configuration.get("single_handle_bridge_algorithm_revision") != (
        SINGLE_HANDLE_BRIDGE_ALGORITHM
    ):
        raise ValueError("CSR was not produced by the reviewed terminal carrier")
    if configuration.get("family_fingerprint") != summary.get("family_fingerprint"):
        raise ValueError("summary and CSR active families disagree")

    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the local CMGDB CSR reader is not installed") from error
    caps = CMGDB.MapGraphCSRCheckpointCaps(
        max_vertices=int(metadata["vertices"]),
        max_edges=int(metadata["edges"]),
        max_payload_bytes=int(metadata["payload_bytes"]),
    )
    graph = CMGDB.load_map_graph_csr_checkpoint(
        relation_path,
        expected_configuration=configuration,
        caps=caps,
        expected_fingerprint=relation_fingerprint,
    )
    if int(graph.num_vertices()) != int(summary["active_cells"]):
        raise ValueError("CSR and acceptance summary vertex counts disagree")

    provenance_path = stage / "provenance.jsonl.gz"
    provenance_reference = validate_spiking_neuron_provenance(
        provenance_path,
        expected_relation_csr_fingerprint=relation_fingerprint,
    )
    records: list[AtlasTopCellRecord] = []
    header: Mapping[str, object] | None = None
    with gzip.open(provenance_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            payload = json.loads(line)
            record_type = payload.get("record")
            if line_number == 0:
                if record_type != "header":
                    raise ValueError("source provenance does not begin with its header")
                header = payload
                continue
            if record_type == "trailer":
                break
            if record_type != "source":
                raise ValueError("unknown source-provenance record")
            index = int(payload["index"])
            if index != len(records):
                raise ValueError("source-provenance indices are not consecutive")
            graph_targets = tuple(
                sorted(int(target) for target in graph.adjacencies(index))
            )
            recorded_targets = tuple(int(target) for target in payload["targets"])
            if recorded_targets != graph_targets:
                raise ValueError(
                    f"source provenance and CSR differ on row {index}"
                )
            failures = Counter(str(reason) for reason in payload.get("failures", ()))
            records.append(
                AtlasTopCellRecord(
                    index=index,
                    chart_id=int(payload["chart_id"]),
                    bounds=tuple(float(value) for value in payload["bounds"]),
                    image=graph_targets,
                    morse_node=(
                        None
                        if payload.get("morse_node") is None
                        else int(payload["morse_node"])
                    ),
                    relation_evaluated=True,
                    callback_failed_samples=sum(failures.values()),
                    callback_failure_reasons=tuple(sorted(failures.items())),
                )
            )
    if header is None or len(records) != int(graph.num_vertices()):
        raise ValueError("source provenance does not cover the complete CSR relation")
    if header.get("stage_gates_passed") is not True:
        raise ValueError("source provenance header is not gate-passing")
    if header.get("relation_csr", {}).get("fingerprint") != relation_fingerprint:
        raise ValueError("source-provenance header is bound to a different CSR")

    morse_nodes = tuple(
        sorted({cell.morse_node for cell in records if cell.morse_node is not None})
    )
    if len(morse_nodes) != int(summary["morse_nodes"]):
        raise ValueError("Morse membership and summary node count disagree")
    bridge_diagnostics = header.get("single_handle_bridge_diagnostics", {})
    snapshot = AtlasRelationSnapshot(
        model="compact_quadratic_integrate_and_fire",
        depth=int(summary["total_depth"]),
        t_star=float(summary["t_star"]),
        cells=tuple(records),
        morse_nodes=morse_nodes,
        morse_edges=tuple(
            (int(source), int(target))
            for source, target in summary.get("morse_edges", ())
        ),
        base_chart_id=0,
        handle_chart_id=1,
        whole_cell_outer_enclosure_certified=False,
        global_attractor_lattice_interpretation=False,
        box_map_failed_samples=sum(
            cell.callback_failed_samples for cell in records
        ),
        box_map_empty_images=sum(not cell.image for cell in records),
        box_map_unresolved_stage_edges=int(
            bridge_diagnostics.get("residual_unresolved_stage_edges", 0)
        ),
        relation_scope="full_atlas",
    )
    return SpikingNeuronConleyInput(
        stage_directory=stage,
        snapshot=snapshot,
        stage_summary=summary,
        relation_reference=dict(relation_reference),
        provenance_reference=provenance_reference,
    )


@dataclass(frozen=True)
class SpikingNeuronFiniteRelationConleyAudit:
    """Computed finite shift class plus exact artifact provenance."""

    input: SpikingNeuronConleyInput
    audit: AtlasNerveFiniteRelationAudit
    pair_fingerprint: str
    nerve_fingerprint: str
    carrier_fingerprint: str
    reference_support: Mapping[str, object]

    @property
    def computed(self) -> bool:
        return self.audit.finite_relation_conley_index_computed

    def checkpoint_payload(self) -> dict[str, object]:
        top_pair = self.audit.top_pair
        payload = {
            "schema": CONLEY_CHECKPOINT_SCHEMA,
            "protocol_revision": CONLEY_PROTOCOL_REVISION,
            "relation_csr_fingerprint": self.input.relation_reference["fingerprint"],
            "source_provenance_trailer_fingerprint": (
                self.input.provenance_reference["trailer_fingerprint"]
            ),
            "pair_fingerprint": self.pair_fingerprint,
            "nerve_fingerprint": self.nerve_fingerprint,
            "carrier_fingerprint": self.carrier_fingerprint,
            "reference_support": dict(self.reference_support),
            "top_cell_pair": {
                "morse_node": top_pair.morse_node,
                "S": sorted(top_pair.s_cells),
                "F_S": sorted(top_pair.f_s_cells),
                "X": sorted(top_pair.x_cells),
                "A": sorted(top_pair.a_cells),
                "raw_A_exit_edges": [
                    [source, list(targets)]
                    for source, targets in top_pair.a_targets_outside_x
                ],
            },
            "cmgdb_relative_homology_payload": (
                self.audit.preparation.payload.to_json_dict(include_basis=True)
            ),
            "finite_relation_shift_class": dict(
                self.audit.finite_relation_shift_class or {}
            ),
            "analytic_conley_label_attached": False,
            "continuous_system_conley_index_certified": False,
        }
        payload["fingerprint"] = _fingerprint(payload)
        return payload

    def summary(self) -> dict[str, object]:
        base = self.audit.summary()
        top_pair = self.audit.top_pair
        carrier_sizes = tuple(
            len(self.audit.preparation.carrier.image(cell))
            for cell in self.audit.pair.complex.cells
        )
        base.update(
            {
                "schema": "spiking-neuron-finite-relation-conley-audit-v1",
                "protocol_revision": CONLEY_PROTOCOL_REVISION,
                "input_artifacts": {
                    "relation_csr": dict(self.input.relation_reference),
                    "source_provenance": dict(self.input.provenance_reference),
                    "family_fingerprint": self.input.stage_summary[
                        "family_fingerprint"
                    ],
                    "acceptance_stage_gates_passed": True,
                },
                "pair_fingerprint": self.pair_fingerprint,
                "nerve_fingerprint": self.nerve_fingerprint,
                "carrier_fingerprint": self.carrier_fingerprint,
                "reference_support": dict(self.reference_support),
                "complete_unpruned_relation_used": True,
                "raw_exit_edges_retained_in_pair_audit": True,
                "raw_exit_edges_from_A": sum(
                    len(targets)
                    for _source, targets in top_pair.a_targets_outside_x
                ),
                "intersections_audited": len(self.audit.nerve.cells),
                "all_nonempty_intersections_contractible": True,
                "carrier_value_cell_count_mean": (
                    sum(carrier_sizes) / len(carrier_sizes)
                ),
                "finite_relation_conley_index_computed": self.computed,
                "coefficient_field": "GF(5)",
                "expected_polynomial_gate_used": False,
                "analytic_fallback_used": False,
                "sampled_terminal_carrier_assumption_used": True,
                "numerically_rigorous_outer_approximation_claimed": False,
                "continuous_system_conley_index_certified": False,
                "continuous_system_blockers": sorted(
                    set(base["continuous_system_blockers"])
                    | {
                        "sampled_box_map_not_a_rigorous_outer_enclosure",
                        "terminal_single_handle_carrier_is_an_explicit_modeling_assumption",
                    }
                ),
            }
        )
        return base


def _nerve_fingerprint(audit: AtlasNerveFiniteRelationAudit) -> str:
    nerve = audit.nerve
    return _fingerprint(
        {
            "metadata": dict(nerve.metadata),
            "simplices": [
                [simplex.dimension, list(simplex.vertices)]
                for simplex in nerve.cells
            ],
        }
    )


def _carrier_fingerprint(audit: AtlasNerveFiniteRelationAudit) -> str:
    digest = hashlib.sha256()
    carrier = audit.preparation.carrier
    for source in audit.pair.complex.cells:
        image = sorted(
            carrier.image(source), key=lambda cell: (cell.dimension, cell.vertices)
        )
        digest.update(
            _canonical_json(
                {
                    "source": [source.dimension, list(source.vertices)],
                    "image": [
                        [cell.dimension, list(cell.vertices)] for cell in image
                    ],
                }
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _reference_source_support(
    loaded: SpikingNeuronConleyInput,
    recurrent_cells: frozenset[int],
    *,
    morse_node: int,
) -> dict[str, object]:
    """Reconstruct and bind every frozen reference-cycle source label.

    This is the same 129 base plus 65 open-handle sampling convention used by
    the acceptance computation.  The exact label-to-Atlas-id map lets a later
    no-ODE audit verify the complete represented reference cycle against the
    authenticated recurrent set.
    """

    cells = tuple(
        AtlasNeuronCell(
            cell.index,
            cell.chart_id,
            tuple(cell.bounds),
        )
        for cell in loaded.snapshot.cells
    )
    cycle = compute_neuron_reference_cycle()
    labels: dict[str, list[int]] = {}
    base_times = np.linspace(
        0.0,
        cycle.flight_time,
        REFERENCE_BASE_SAMPLES,
        endpoint=False,
    )
    for index, base_time in enumerate(base_times):
        state = np.asarray(cycle.solution(float(base_time)), dtype=np.float64)
        labels[f"reference-base-{index}"] = sorted(
            cell.index
            for cell in _cells_containing_point(cells, BASE_CHART_ID, state)
        )
    charts = spiking_neuron_atlas_charts()
    guard = np.asarray((35.0, cycle.pre_reset_u), dtype=np.float64)
    phases = np.linspace(0.0, 1.0, REFERENCE_HANDLE_SAMPLES + 2)[1:-1]
    for index, phase in enumerate(phases):
        coordinates = charts.encode_handle(guard, float(phase))
        labels[f"reference-handle-{index}"] = sorted(
            cell.index
            for cell in _cells_containing_point(
                cells, HANDLE_CHART_ID, coordinates
            )
        )

    recurrent_labels = {
        label: [source for source in sources if source in recurrent_cells]
        for label, sources in labels.items()
    }
    uncovered = sorted(
        label for label, sources in recurrent_labels.items() if not sources
    )
    if uncovered:
        raise ValueError(
            "frozen reference cycle is not completely represented in S: "
            f"{uncovered[:10]!r}"
        )
    recurrent_hits = sum(len(sources) for sources in recurrent_labels.values())
    declared_hits = loaded.stage_summary.get("reference_node_hits", {}).get(
        str(int(morse_node)), 0
    )
    if recurrent_hits != int(declared_hits):
        raise ValueError(
            "reconstructed reference source hits disagree with acceptance summary"
        )
    support = {
        "protocol": "129 base clocks plus 65 open-handle phases",
        "base_samples": REFERENCE_BASE_SAMPLES,
        "handle_samples": REFERENCE_HANDLE_SAMPLES,
        "labels": labels,
        "recurrent_label_sources": recurrent_labels,
        "source_ids": sorted({source for sources in labels.values() for source in sources}),
        "recurrent_source_ids": sorted(
            {source for sources in recurrent_labels.values() for source in sources}
        ),
        "source_hits_with_multiplicity": sum(len(sources) for sources in labels.values()),
        "recurrent_hits_with_multiplicity": recurrent_hits,
        "all_labels_intersect_recurrent_pair": True,
        "relation_csr_fingerprint": loaded.relation_reference["fingerprint"],
    }
    support["fingerprint"] = _fingerprint(support)
    return support


def compute_spiking_neuron_finite_relation_conley(
    stage_directory: str | Path,
    *,
    morse_node: int = 0,
) -> SpikingNeuronFiniteRelationConleyAudit:
    """Compute the standard-pair finite index on the strict reset quotient."""

    loaded = load_spiking_neuron_conley_input(stage_directory)
    audit = audit_atlas_nerve_finite_relation(
        loaded.snapshot,
        morse_node=morse_node,
        candidate_name="t20_reference_recurrent_component",
        gluing=spiking_neuron_atlas_reset_gluing(),
        interior_seam_subcomplexes=("reset",),
    )
    if audit.finite_relation_blockers:
        raise ValueError(
            "finite-relation Conley gates failed: "
            f"{list(audit.finite_relation_blockers)!r}"
        )
    if not audit.finite_relation_conley_index_computed:
        raise AssertionError("finite-relation shift class was not computed")
    top_pair = audit.top_pair
    reference_support = _reference_source_support(
        loaded,
        top_pair.s_cells,
        morse_node=top_pair.morse_node,
    )
    pair_fingerprint = _fingerprint(
        {
            "morse_node": top_pair.morse_node,
            "S": sorted(top_pair.s_cells),
            "F_S": sorted(top_pair.f_s_cells),
            "X": sorted(top_pair.x_cells),
            "A": sorted(top_pair.a_cells),
            "raw_A_exit_edges": [
                [source, list(targets)]
                for source, targets in top_pair.a_targets_outside_x
            ],
        }
    )
    return SpikingNeuronFiniteRelationConleyAudit(
        input=loaded,
        audit=audit,
        pair_fingerprint=pair_fingerprint,
        nerve_fingerprint=_nerve_fingerprint(audit),
        carrier_fingerprint=_carrier_fingerprint(audit),
        reference_support=reference_support,
    )


def write_spiking_neuron_conley_checkpoint(
    result: SpikingNeuronFiniteRelationConleyAudit,
    path: str | Path,
) -> dict[str, object]:
    """Atomically persist the exact sparse complex/map payload and bindings."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite Conley checkpoint {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    payload = result.checkpoint_payload()
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_stream,
                mtime=0,
            ) as stream:
                stream.write(_canonical_json(payload) + b"\n")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    reference = validate_spiking_neuron_conley_checkpoint(
        target,
        expected_relation_csr_fingerprint=str(
            result.input.relation_reference["fingerprint"]
        ),
    )
    return reference


def validate_spiking_neuron_conley_checkpoint(
    path: str | Path,
    *,
    expected_relation_csr_fingerprint: str | None = None,
) -> dict[str, object]:
    """Strictly reload and independently recompute the stored GF(5) result."""

    source = Path(path)
    with gzip.open(source, "rb") as stream:
        raw = stream.read()
    payload = json.loads(raw)
    if raw != _canonical_json(payload) + b"\n":
        raise ValueError("Conley checkpoint JSON is not canonical")
    if payload.get("schema") != CONLEY_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported spiking-neuron Conley checkpoint")
    if payload.get("protocol_revision") != CONLEY_PROTOCOL_REVISION:
        raise ValueError("unsupported spiking-neuron Conley protocol revision")
    if payload.get("analytic_conley_label_attached") is not False:
        raise ValueError("Conley checkpoint unexpectedly attaches an analytic label")
    if payload.get("continuous_system_conley_index_certified") is not False:
        raise ValueError("Conley checkpoint overstates continuous certification")
    claimed = payload.get("fingerprint")
    unsigned = dict(payload)
    unsigned.pop("fingerprint", None)
    if claimed != _fingerprint(unsigned):
        raise ValueError("Conley checkpoint fingerprint mismatch")
    relation_fingerprint = payload.get("relation_csr_fingerprint")
    if (
        expected_relation_csr_fingerprint is not None
        and relation_fingerprint != expected_relation_csr_fingerprint
    ):
        raise ValueError("Conley checkpoint is bound to a different CSR relation")

    top_pair = payload.get("top_cell_pair")
    if not isinstance(top_pair, Mapping):
        raise ValueError("Conley checkpoint lacks its exact top-cell pair")
    if payload.get("pair_fingerprint") != _fingerprint(top_pair):
        raise ValueError("Conley checkpoint top-cell pair fingerprint mismatch")
    s_cells = {int(value) for value in top_pair["S"]}
    f_s_cells = {int(value) for value in top_pair["F_S"]}
    x_cells = {int(value) for value in top_pair["X"]}
    a_cells = {int(value) for value in top_pair["A"]}
    if any(
        len(values) != len(set(int(value) for value in values))
        for values in (
            top_pair["S"],
            top_pair["F_S"],
            top_pair["X"],
            top_pair["A"],
        )
    ):
        raise ValueError("Conley checkpoint top-cell pair contains duplicate ids")
    if x_cells != s_cells | f_s_cells or a_cells != x_cells - s_cells:
        raise ValueError("Conley checkpoint does not satisfy X=S union F(S), A=X minus S")
    if s_cells & a_cells:
        raise ValueError("Conley checkpoint S and A are not disjoint")
    reference_support = payload.get("reference_support")
    if not isinstance(reference_support, Mapping):
        raise ValueError("Conley checkpoint lacks its exact reference support")
    unsigned_support = dict(reference_support)
    support_fingerprint = unsigned_support.pop("fingerprint", None)
    if support_fingerprint != _fingerprint(unsigned_support):
        raise ValueError("Conley checkpoint reference-support fingerprint mismatch")
    if reference_support.get("relation_csr_fingerprint") != relation_fingerprint:
        raise ValueError("reference support is bound to a different CSR relation")
    reference_ids = {int(value) for value in reference_support["source_ids"]}
    recurrent_reference_ids = {
        int(value) for value in reference_support["recurrent_source_ids"]
    }
    if not recurrent_reference_ids <= reference_ids:
        raise ValueError("recurrent reference support is not a subset of all support")
    if not reference_ids <= s_cells:
        raise ValueError("persisted reference source support is not contained in S")
    labels = reference_support.get("labels")
    recurrent_labels = reference_support.get("recurrent_label_sources")
    if not isinstance(labels, Mapping) or not isinstance(recurrent_labels, Mapping):
        raise ValueError("reference support lacks its exact label maps")
    base_samples = int(reference_support["base_samples"])
    handle_samples = int(reference_support["handle_samples"])
    expected_labels = {
        *(f"reference-base-{index}" for index in range(base_samples)),
        *(f"reference-handle-{index}" for index in range(handle_samples)),
    }
    if set(labels) != expected_labels or set(recurrent_labels) != expected_labels:
        raise ValueError("reference support labels do not match its frozen protocol")
    normalized_labels = {
        str(label): [int(value) for value in values]
        for label, values in labels.items()
    }
    if any(not values for values in normalized_labels.values()):
        raise ValueError("reference support contains an empty source label")
    normalized_recurrent = {
        str(label): [int(value) for value in values]
        for label, values in recurrent_labels.items()
    }
    expected_recurrent = {
        label: [value for value in values if value in s_cells]
        for label, values in normalized_labels.items()
    }
    if normalized_recurrent != expected_recurrent:
        raise ValueError("recurrent reference label map is inconsistent with S")
    if any(not values for values in normalized_recurrent.values()):
        raise ValueError("a reference label does not intersect recurrent S")
    if reference_ids != {
        value for values in normalized_labels.values() for value in values
    }:
        raise ValueError("reference source_ids are not the exact label union")
    if recurrent_reference_ids != {
        value for values in normalized_recurrent.values() for value in values
    }:
        raise ValueError("recurrent_source_ids are not the exact recurrent label union")
    if int(reference_support["source_hits_with_multiplicity"]) != sum(
        len(values) for values in normalized_labels.values()
    ):
        raise ValueError("reference source-hit multiplicity is inconsistent")
    if int(reference_support["recurrent_hits_with_multiplicity"]) != sum(
        len(values) for values in normalized_recurrent.values()
    ):
        raise ValueError("recurrent reference-hit multiplicity is inconsistent")
    if reference_support.get("all_labels_intersect_recurrent_pair") is not True:
        raise ValueError("persisted reference labels do not all intersect S")

    stored = payload["cmgdb_relative_homology_payload"]
    homology_payload = CMGDBRelativeHomologyPayload(
        cell_counts=tuple(int(value) for value in stored["cell_counts"]),
        boundary_entries=tuple(
            tuple(tuple(int(value) for value in entry) for entry in degree)
            for degree in stored["boundary_entries"]
        ),
        chain_map_entries=tuple(
            tuple(tuple(int(value) for value in entry) for entry in degree)
            for degree in stored["chain_map_entries"]
        ),
        basis_by_dimension=tuple(
            tuple(str(value) for value in degree)
            for degree in stored["basis_by_dimension"]
        ),
        coefficient_field=int(stored["coefficient_field"]),
    )
    try:
        import CMGDB
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("the local CMGDB homology bridge is not installed") from error
    recomputed = dict(
        CMGDB.ComputeRelativeHomologyShiftClass(
            *homology_payload.as_compute_args()
        )
    )
    expected = dict(payload["finite_relation_shift_class"])
    for key in (
        "cell_counts",
        "coefficient_field",
        "homology_dimensions",
        "induced_maps",
        "shift_class",
        "validation",
    ):
        if recomputed.get(key) != expected.get(key):
            raise ValueError(f"stored and recomputed Conley results differ at {key}")
    return {
        "schema": "spiking-neuron-conley-chain-reference-v1",
        "path": source.name,
        "fingerprint": claimed,
        "file_sha256": _sha256_file(source),
        "relation_csr_fingerprint": relation_fingerprint,
        "pair_fingerprint": payload["pair_fingerprint"],
        "nerve_fingerprint": payload["nerve_fingerprint"],
        "carrier_fingerprint": payload["carrier_fingerprint"],
        "reference_support_fingerprint": support_fingerprint,
        "reference_source_ids": list(
            payload["reference_support"]["source_ids"]
        ),
        "finite_relation_shift_class": list(expected["shift_class"]),
        "finite_relation_conley_index": expected,
        "cell_counts": list(homology_payload.cell_counts),
        "boundary_nonzero_entries": [
            len(entries) for entries in homology_payload.boundary_entries
        ],
        "chain_map_nonzero_entries": [
            len(entries) for entries in homology_payload.chain_map_entries
        ],
        "strict_reload_recomputed_shift_class": True,
    }


def validate_spiking_neuron_conley_summary(
    path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    expected_relation_csr_fingerprint: str | None = None,
) -> dict[str, object]:
    """Validate the summary self-hash and every checkpoint binding."""

    source = Path(path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    if summary.get("schema") != "spiking-neuron-finite-relation-conley-audit-v1":
        raise ValueError("unsupported spiking-neuron Conley summary")
    claimed = summary.get("fingerprint")
    unsigned = dict(summary)
    unsigned.pop("fingerprint", None)
    if claimed != _fingerprint(unsigned):
        raise ValueError("Conley summary fingerprint mismatch")
    checkpoint_reference = summary.get("chain_checkpoint")
    if not isinstance(checkpoint_reference, Mapping):
        raise ValueError("Conley summary lacks its chain checkpoint reference")
    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else source.parent / str(checkpoint_reference["path"])
    )
    relation_fingerprint = str(
        summary["input_artifacts"]["relation_csr"]["fingerprint"]
    )
    if (
        expected_relation_csr_fingerprint is not None
        and relation_fingerprint != expected_relation_csr_fingerprint
    ):
        raise ValueError("Conley summary is bound to a different CSR relation")
    validated = validate_spiking_neuron_conley_checkpoint(
        resolved_checkpoint,
        expected_relation_csr_fingerprint=relation_fingerprint,
    )
    for key in (
        "fingerprint",
        "file_sha256",
        "relation_csr_fingerprint",
        "pair_fingerprint",
        "nerve_fingerprint",
        "carrier_fingerprint",
        "cell_counts",
        "boundary_nonzero_entries",
        "chain_map_nonzero_entries",
    ):
        if checkpoint_reference.get(key) != validated.get(key):
            raise ValueError(
                f"Conley summary and checkpoint reference differ at {key}"
            )
    for key in ("pair_fingerprint", "nerve_fingerprint", "carrier_fingerprint"):
        if summary.get(key) != validated.get(key):
            raise ValueError(f"Conley summary and checkpoint differ at {key}")
    if summary["reference_support"]["fingerprint"] != validated[
        "reference_support_fingerprint"
    ]:
        raise ValueError("Conley summary and checkpoint reference support differ")
    if summary.get("finite_relation_shift_class") != validated[
        "finite_relation_shift_class"
    ]:
        raise ValueError("Conley summary and checkpoint shift classes differ")
    if summary.get("finite_relation_conley_index") != validated[
        "finite_relation_conley_index"
    ]:
        raise ValueError("Conley summary and checkpoint full finite results differ")
    if not summary.get("finite_relation_conley_index_computed"):
        raise ValueError("Conley summary does not declare a computed finite index")
    if summary.get("finite_relation_blockers"):
        raise ValueError("Conley summary retains finite-relation blockers")
    if summary.get("continuous_system_conley_index_certified") is not False:
        raise ValueError("Conley summary overstates continuous-system certification")
    if summary.get("analytic_conley_label_attached") is not False:
        raise ValueError("Conley summary unexpectedly attaches an analytic label")
    if summary.get("whole_cell_outer_enclosure_certified") is not False:
        raise ValueError("Conley summary overstates whole-cell enclosure certification")
    if summary.get("numerically_rigorous_outer_approximation_claimed") is not False:
        raise ValueError("Conley summary overstates numerical rigor")
    return {
        "schema": "spiking-neuron-conley-summary-reference-v1",
        "path": source.name,
        "fingerprint": claimed,
        "relation_csr_fingerprint": relation_fingerprint,
        "checkpoint": validated,
        "strict_summary_and_checkpoint_bindings_validated": True,
    }


__all__ = [
    "CONLEY_CHECKPOINT_SCHEMA",
    "CONLEY_PROTOCOL_REVISION",
    "SpikingNeuronConleyInput",
    "SpikingNeuronFiniteRelationConleyAudit",
    "compute_spiking_neuron_finite_relation_conley",
    "load_spiking_neuron_conley_input",
    "validate_spiking_neuron_conley_checkpoint",
    "validate_spiking_neuron_conley_summary",
    "write_spiking_neuron_conley_checkpoint",
]
