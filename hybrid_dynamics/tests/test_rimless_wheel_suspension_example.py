"""Regression tests for the physical rimless-wheel suspension pipeline."""

import numpy as np

from hybrid_dynamics.examples.rimless_wheel_suspension import (
    build_rimless_wheel_suspension_pipeline,
    stance_energy,
    walking_fixed_point_speed,
)
from hybrid_dynamics.src.sampled_suspension import BaseSuspensionSample


def test_walking_speed_closes_the_impact_map_and_preserves_stance_energy():
    speed = walking_fixed_point_speed(alpha=0.4, gamma=0.2)
    assert 0.5 < speed < 0.6
    assert stance_energy(np.array([-0.2, speed])) > 1.0

    result = build_rimless_wheel_suspension_pipeline()

    assert result.diagnostics["stance_energy_residual"] < 1e-9
    assert result.diagnostics["reset_fixed_point_residual"] < 1e-9
    assert result.diagnostics["guard_transverse_speed"] > 0.0
    assert isinstance(result.witnesses[0].endpoint, BaseSuspensionSample)


def test_rimless_relation_and_circle_skeleton_are_materialized():
    result = build_rimless_wheel_suspension_pipeline()

    assert result.relation_graph.number_of_edges() > 0
    assert len(result.phase_cells) == result.handle_slabs
    assert len(result.phase_descriptors) == 1
    assert result.descriptor_expansion_matches_relation
    assert len(result.scc_reconstruction.components) == 1
    assert len(result.recurrent_sccs) == 1
    assert result.suspension_complex.betti_numbers(modulus=5) == (1, 1)
    assert result.fixed_time_carrier.carries(result.cellular_chain_map)
    counts, boundaries, chain_map = result.cmgdb_payload.as_compute_args()
    assert counts == list(result.relative_pair.cell_counts)
    assert len(boundaries) == len(chain_map) == 2
