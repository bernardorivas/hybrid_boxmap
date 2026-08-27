"""Regression tests for the physical bouncing-ball suspension pipeline."""

import numpy as np

from hybrid_dynamics.examples.bouncing_ball_suspension import (
    analytic_next_impact_time,
    build_bouncing_ball_suspension_pipeline,
)
from hybrid_dynamics.src.sampled_suspension import (
    BaseCell,
    HandleSuspensionSample,
)


def test_ballistic_impact_and_zeno_unit_handle_are_computed():
    expected = np.sqrt(2.0 / 9.81)
    assert analytic_next_impact_time(np.array([1.0, 0.0])) == expected

    result = build_bouncing_ball_suspension_pipeline()

    assert result.model_name == "bouncing_ball_zeno_suspension"
    assert result.t_star == 0.5
    assert result.diagnostics["impact_time_absolute_error"] < 1e-9
    falling, rest = result.witnesses
    assert isinstance(falling.endpoint, HandleSuspensionSample)
    assert isinstance(rest.endpoint, HandleSuspensionSample)
    assert abs(falling.endpoint.phase - (0.5 - expected)) < 1e-8
    assert abs(rest.endpoint.phase - 0.5) < 1e-12


def test_bouncing_ball_relation_and_circle_skeleton_are_materialized():
    result = build_bouncing_ball_suspension_pipeline()

    assert len(result.phase_cells) == result.handle_slabs
    assert len(result.phase_descriptors) == 1
    assert result.descriptor_expansion_matches_relation
    assert len(result.scc_reconstruction.components) == 1
    assert len(result.recurrent_sccs) == 1
    assert any(isinstance(cell, BaseCell) for cell in result.recurrent_sccs[0])
    assert result.suspension_complex.betti_numbers(modulus=5) == (1, 1)
    assert result.fixed_time_carrier.carries(result.cellular_chain_map)
    assert result.cmgdb_payload.cell_counts == result.relative_pair.cell_counts
    assert result.conley_index_status.startswith("not computed")
