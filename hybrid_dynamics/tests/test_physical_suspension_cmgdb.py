"""Integration regression for all physical suspension examples and CMGDB."""

import json
from pathlib import Path

import pytest

from hybrid_dynamics.examples.physical_suspension_examples import (
    run_all_physical_suspension_examples,
)


def test_all_physical_skeletons_reach_the_generalized_cmgdb_endpoint():
    cmgdb = pytest.importorskip("CMGDB")
    if not hasattr(cmgdb, "ComputeRelativeHomologyShiftClass"):
        pytest.skip("the installed CMGDB does not include the finite-complex bridge")

    results = run_all_physical_suspension_examples(compute_cmgdb=True)
    baseline = json.loads(
        (Path(__file__).parents[2] / "physical_suspension_baseline.json").read_text(),
    )

    assert len(results) == 3
    assert len(baseline["examples"]) == 3
    for result, expected in zip(results, baseline["examples"], strict=True):
        assert result.descriptor_expansion_matches_relation
        assert result.model_name == expected["model"]
        assert result.relation_graph.number_of_nodes() == expected["relation_nodes"]
        assert result.relation_graph.number_of_edges() == expected["relation_edges"]
        assert result.relation_hash == expected["relation_hash"]
        assert [len(component) for component in result.recurrent_sccs] == expected[
            "recurrent_scc_sizes"
        ]
        assert result.cmgdb_result is not None
        assert result.cmgdb_result["validation"] == {
            "matrix_shapes_and_entries": True,
            "boundary_squared_zero": True,
            "chain_map_equation": True,
        }
        assert result.cmgdb_result["homology_dimensions"] == [1, 1]
        assert result.cmgdb_result["induced_maps"] == [[[1]], [[1]]]
        assert result.cmgdb_result["shift_class"] == ["x-1", "x-1"]
        assert "not a Conley-index claim" in result.conley_index_status
