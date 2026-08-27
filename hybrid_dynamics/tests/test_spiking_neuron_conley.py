from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import hybrid_dynamics.examples.spiking_neuron_conley as neuron_conley
from demo.plot_spiking_neuron_finite_relation import _morse_rectangles


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    with gzip.open(path, "wb") as stream:
        stream.write(neuron_conley._canonical_json(payload) + b"\n")  # noqa: SLF001


def _minimal_checkpoint_payload() -> dict[str, object]:
    import CMGDB

    relation_fingerprint = "a" * 64
    top_pair = {
        "morse_node": 0,
        "S": [0],
        "F_S": [0],
        "X": [0],
        "A": [],
        "raw_A_exit_edges": [],
    }
    support = {
        "protocol": "synthetic",
        "base_samples": 1,
        "handle_samples": 0,
        "labels": {"reference-base-0": [0]},
        "recurrent_label_sources": {"reference-base-0": [0]},
        "source_ids": [0],
        "recurrent_source_ids": [0],
        "source_hits_with_multiplicity": 1,
        "recurrent_hits_with_multiplicity": 1,
        "all_labels_intersect_recurrent_pair": True,
        "relation_csr_fingerprint": relation_fingerprint,
    }
    support["fingerprint"] = neuron_conley._fingerprint(support)  # noqa: SLF001
    finite = dict(
        CMGDB.ComputeRelativeHomologyShiftClass(
            [1],
            [[]],
            [[(0, 0, 1)]],
        )
    )
    finite.update(
        {
            "result_scope": "finite_reset_quotient_relation",
            "continuous_system_conley_index_certified": False,
            "finite_relation_algebra_validated": True,
        }
    )
    payload = {
        "schema": neuron_conley.CONLEY_CHECKPOINT_SCHEMA,
        "protocol_revision": neuron_conley.CONLEY_PROTOCOL_REVISION,
        "relation_csr_fingerprint": relation_fingerprint,
        "source_provenance_trailer_fingerprint": "b" * 64,
        "pair_fingerprint": neuron_conley._fingerprint(top_pair),  # noqa: SLF001
        "nerve_fingerprint": "c" * 64,
        "carrier_fingerprint": "d" * 64,
        "reference_support": support,
        "top_cell_pair": top_pair,
        "cmgdb_relative_homology_payload": {
            "coefficient_field": 5,
            "cell_counts": [1],
            "boundary_entries": [[]],
            "chain_map_entries": [[[0, 0, 1]]],
            "basis_by_dimension": [["vertex-0"]],
        },
        "finite_relation_shift_class": finite,
        "analytic_conley_label_attached": False,
        "continuous_system_conley_index_certified": False,
    }
    payload["fingerprint"] = neuron_conley._fingerprint(payload)  # noqa: SLF001
    return payload


def test_strict_checkpoint_reload_recomputes_finite_shift(tmp_path: Path) -> None:
    path = tmp_path / "chain.json.gz"
    payload = _minimal_checkpoint_payload()
    _write_checkpoint(path, payload)
    validated = neuron_conley.validate_spiking_neuron_conley_checkpoint(
        path,
        expected_relation_csr_fingerprint="a" * 64,
    )
    assert validated["strict_reload_recomputed_shift_class"] is True
    assert validated["finite_relation_shift_class"] == ["x-1"]
    assert validated["reference_source_ids"] == [0]


def test_checkpoint_rejects_rehashed_reference_support_outside_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chain.json.gz"
    payload = _minimal_checkpoint_payload()
    support = dict(payload["reference_support"])
    support["source_ids"] = [1]
    support["recurrent_source_ids"] = [1]
    support.pop("fingerprint")
    support["fingerprint"] = neuron_conley._fingerprint(support)  # noqa: SLF001
    payload["reference_support"] = support
    payload.pop("fingerprint")
    payload["fingerprint"] = neuron_conley._fingerprint(payload)  # noqa: SLF001
    _write_checkpoint(path, payload)
    with pytest.raises(ValueError, match="not contained in S"):
        neuron_conley.validate_spiking_neuron_conley_checkpoint(path)


def test_checkpoint_rejects_rehashed_empty_reference_label(tmp_path: Path) -> None:
    path = tmp_path / "chain.json.gz"
    payload = _minimal_checkpoint_payload()
    support = dict(payload["reference_support"])
    support["labels"] = {"reference-base-0": []}
    support["recurrent_label_sources"] = {"reference-base-0": []}
    support["source_ids"] = []
    support["recurrent_source_ids"] = []
    support["source_hits_with_multiplicity"] = 0
    support["recurrent_hits_with_multiplicity"] = 0
    support.pop("fingerprint")
    support["fingerprint"] = neuron_conley._fingerprint(support)  # noqa: SLF001
    payload["reference_support"] = support
    payload.pop("fingerprint")
    payload["fingerprint"] = neuron_conley._fingerprint(payload)  # noqa: SLF001
    _write_checkpoint(path, payload)
    with pytest.raises(ValueError, match="empty source label"):
        neuron_conley.validate_spiking_neuron_conley_checkpoint(path)


def test_checkpoint_rejects_rehashed_nonstandard_pair_and_claim_flags(
    tmp_path: Path,
) -> None:
    pair_path = tmp_path / "bad-pair.json.gz"
    payload = _minimal_checkpoint_payload()
    top_pair = dict(payload["top_cell_pair"])
    top_pair["X"] = [0, 1]
    payload["top_cell_pair"] = top_pair
    payload["pair_fingerprint"] = neuron_conley._fingerprint(top_pair)  # noqa: SLF001
    payload.pop("fingerprint")
    payload["fingerprint"] = neuron_conley._fingerprint(payload)  # noqa: SLF001
    _write_checkpoint(pair_path, payload)
    with pytest.raises(ValueError, match="X=S union F"):
        neuron_conley.validate_spiking_neuron_conley_checkpoint(pair_path)

    flag_path = tmp_path / "bad-flag.json.gz"
    payload = _minimal_checkpoint_payload()
    payload["analytic_conley_label_attached"] = True
    payload.pop("fingerprint")
    payload["fingerprint"] = neuron_conley._fingerprint(payload)  # noqa: SLF001
    _write_checkpoint(flag_path, payload)
    with pytest.raises(ValueError, match="analytic label"):
        neuron_conley.validate_spiking_neuron_conley_checkpoint(flag_path)


def test_summary_validator_checks_self_hash_and_all_checkpoint_bindings(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "chain.json.gz"
    payload = _minimal_checkpoint_payload()
    _write_checkpoint(checkpoint_path, payload)
    checkpoint = neuron_conley.validate_spiking_neuron_conley_checkpoint(
        checkpoint_path
    )
    summary = {
        "schema": "spiking-neuron-finite-relation-conley-audit-v1",
        "input_artifacts": {
            "relation_csr": {"fingerprint": "a" * 64},
        },
        "chain_checkpoint": checkpoint,
        "pair_fingerprint": payload["pair_fingerprint"],
        "nerve_fingerprint": payload["nerve_fingerprint"],
        "carrier_fingerprint": payload["carrier_fingerprint"],
        "reference_support": payload["reference_support"],
        "finite_relation_shift_class": ["x-1"],
        "finite_relation_conley_index": payload[
            "finite_relation_shift_class"
        ],
        "finite_relation_conley_index_computed": True,
        "finite_relation_blockers": [],
        "continuous_system_conley_index_certified": False,
        "analytic_conley_label_attached": False,
        "whole_cell_outer_enclosure_certified": False,
        "numerically_rigorous_outer_approximation_claimed": False,
    }
    summary["fingerprint"] = neuron_conley._fingerprint(summary)  # noqa: SLF001
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    validated = neuron_conley.validate_spiking_neuron_conley_summary(summary_path)
    assert validated["strict_summary_and_checkpoint_bindings_validated"] is True

    tampered = dict(summary)
    tampered["carrier_fingerprint"] = "e" * 64
    summary_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="summary fingerprint mismatch"):
        neuron_conley.validate_spiking_neuron_conley_summary(summary_path)

    tampered = json.loads(json.dumps(summary))
    tampered["finite_relation_conley_index"]["induced_maps"] = [[]]
    tampered.pop("fingerprint")
    tampered["fingerprint"] = neuron_conley._fingerprint(tampered)  # noqa: SLF001
    summary_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="full finite results differ"):
        neuron_conley.validate_spiking_neuron_conley_summary(summary_path)


def test_persisted_neuron_conley_artifact_strict_no_dynamics_reload() -> None:
    root = Path(__file__).resolve().parents[2]
    stage = root / (
        "data/spiking_neuron_atlas/scientific_clock_v1/"
        "adaptive_terminal_bridge_v1/t20/conley_v2"
    )
    if not stage.exists():
        pytest.skip("persisted neuron Conley artifact is not installed")
    validated = neuron_conley.validate_spiking_neuron_conley_summary(
        stage / "finite_relation_conley.json"
    )
    checkpoint = validated["checkpoint"]
    assert checkpoint["cell_counts"] == [1113, 3934, 3789, 994, 31, 5]
    assert checkpoint["finite_relation_shift_class"] == [
        "x-1",
        "x-1",
        "0",
        "0",
        "0",
        "0",
    ]
    assert len(checkpoint["reference_source_ids"]) == 95


def test_persisted_samples5_conley_and_comparison_strict_no_dynamics_reload(
) -> None:
    root = Path(__file__).resolve().parents[2]
    scientific = root / "data/spiking_neuron_atlas/scientific_clock_v1"
    stage = scientific / (
        "adaptive_terminal_bridge_samples5_v1/t20/conley_v2"
    )
    comparison_path = scientific / (
        "adaptive_terminal_bridge_samples5_v1/comparison.json"
    )
    if not stage.exists() or not comparison_path.exists():
        pytest.skip("persisted neuron samples5 artifacts are not installed")

    validated = neuron_conley.validate_spiking_neuron_conley_summary(
        stage / "finite_relation_conley.json"
    )
    checkpoint = validated["checkpoint"]
    assert checkpoint["cell_counts"] == [1113, 3934, 3789, 994, 31, 5]
    assert checkpoint["finite_relation_shift_class"] == [
        "x-1",
        "x-1",
        "0",
        "0",
        "0",
        "0",
    ]
    assert len(checkpoint["reference_source_ids"]) == 95

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    claimed = comparison.pop("fingerprint")
    assert neuron_conley._fingerprint(comparison) == claimed  # noqa: SLF001
    assert comparison["sampling_persistence_passed"] is True
    assert comparison["continuous_system_conley_index_certified"] is False
    difference = comparison["relation_difference"]
    assert difference["changed_source_rows"] == 68
    assert difference["samples5_edges_removed"] == 217
    assert difference["samples5_edges_added"] == 4831
    assert len(difference["changed_source_indices"]) == 68


def test_figure_rectangles_are_bound_to_authenticated_primary_provenance() -> None:
    root = Path(__file__).resolve().parents[2]
    stage = root / (
        "data/spiking_neuron_atlas/scientific_clock_v1/"
        "adaptive_terminal_bridge_v1/t20"
    )
    summary_path = stage / "conley_v2/finite_relation_conley.json"
    if not summary_path.exists():
        pytest.skip("persisted neuron Conley artifact is not installed")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    relation = summary["input_artifacts"]["relation_csr"]
    provenance = summary["input_artifacts"]["source_provenance"]

    rectangles, authenticated = _morse_rectangles(
        stage / "provenance.jsonl.gz",
        expected_relation_csr_fingerprint=relation["fingerprint"],
        expected_source_count=relation["vertices"],
        expected_provenance_reference=provenance,
    )
    assert {chart: len(boxes) for chart, boxes in rectangles.items()} == {
        0: 345,
        1: 768,
    }
    assert authenticated == provenance

    with pytest.raises(ValueError, match="different CSR relation"):
        _morse_rectangles(
            stage / "provenance.jsonl.gz",
            expected_relation_csr_fingerprint="0" * 64,
            expected_source_count=relation["vertices"],
            expected_provenance_reference=provenance,
        )

    wrong_reference = dict(provenance)
    wrong_reference["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="strict Conley input reference"):
        _morse_rectangles(
            stage / "provenance.jsonl.gz",
            expected_relation_csr_fingerprint=relation["fingerprint"],
            expected_source_count=relation["vertices"],
            expected_provenance_reference=wrong_reference,
        )
