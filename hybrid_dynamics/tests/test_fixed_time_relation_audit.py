from __future__ import annotations

import numpy as np
import pytest

from hybrid_dynamics.src.fixed_time_relation_audit import (
    AugmentedQuotientIncidence,
    EndpointProbe,
    RelationAuditFailure,
    audit_dense_base_endpoints,
    audit_endpoint_probes,
    audit_relation_image_connectivity,
)
from hybrid_dynamics.src.fixed_time_suspension_grid import (
    GridResetHandle,
    GridSuspensionIngredients,
)
from hybrid_dynamics.src.grid import Grid
from hybrid_dynamics.src.sampled_suspension import (
    BaseCell,
    BaseSuspensionSample,
    HandleSuspensionSample,
    PhaseCell,
)


def _base_sample(value: float) -> BaseSuspensionSample:
    return BaseSuspensionSample(
        state=np.asarray([value], dtype=float),
        total_time=0.5,
        continuous_time=0.5,
        jumps_completed=0,
    )


def _translation_to_unit_handle(point: np.ndarray):
    """Exact endpoint for x'=1, G={1}, r(1)=0, suspension time 1/2."""

    phase = float(point[0] - 0.5)
    if phase <= 1e-12:
        return _base_sample(1.0)
    return HandleSuspensionSample(
        guard_state=np.asarray([1.0]),
        reset_state=np.asarray([0.0]),
        phase=phase,
        total_time=0.5,
        continuous_time=float(1.0 - point[0]),
        jump_index=0,
    )


def test_dense_probe_exposes_corner_only_missed_handle_segment() -> None:
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[2])
    source = BaseCell(1)  # [1/2, 1]
    # The two source corners give the guard face and phase 1/2.  With one slab
    # of phase padding around phase 1/2, the old rule records slabs 6--9 but
    # misses the true handle segment at phases near 1/4.
    relation = {
        source: frozenset(
            {BaseCell(1), *(PhaseCell(1, slab) for slab in range(6, 10))}
        )
    }

    corner_audit = audit_endpoint_probes(
        relation,
        (
            EndpointProbe(source, _translation_to_unit_handle(np.asarray([0.5]))),
            EndpointProbe(source, _translation_to_unit_handle(np.asarray([1.0]))),
        ),
        grid,
        handle_slabs=16,
    )
    assert corner_audit.passed

    dense_audit = audit_dense_base_endpoints(
        relation,
        grid,
        _translation_to_unit_handle,
        handle_slabs=16,
        source_cells=[1],
        subdivision_level=2,
    )
    assert not dense_audit.passed
    assert dense_audit.missed
    assert any(
        witness.probe.initial_state == (0.75,)
        for witness in dense_audit.missed
    )
    with pytest.raises(RelationAuditFailure, match="missed endpoints"):
        dense_audit.require_passed(context="translation benchmark")


def _ingredients() -> GridSuspensionIngredients:
    phase_cells = tuple(PhaseCell(1, slab) for slab in range(16))
    handle = GridResetHandle(
        handle_id="reset",
        guard_cell=BaseCell(1),
        reset_cells=(BaseCell(0),),
        phase_cells=phase_cells,
        guard_state=(1.0,),
        reset_state=(0.0,),
    )
    return GridSuspensionIngredients(
        base_bounds=((0.0, 1.0),),
        base_subdivisions=(2,),
        handles=(handle,),
        phase_slabs=16,
    )


def test_quotient_incidence_uses_both_handle_gluings() -> None:
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[2])
    incidence = AugmentedQuotientIncidence(grid, _ingredients())

    assert incidence.intersects(BaseCell(1), PhaseCell(1, 0))
    assert incidence.intersects(BaseCell(0), PhaseCell(1, 15))
    assert not incidence.intersects(BaseCell(0), PhaseCell(1, 7))

    assert len(
        incidence.connected_components({BaseCell(0), PhaseCell(1, 15)})
    ) == 1
    assert len(
        incidence.connected_components({BaseCell(0), PhaseCell(1, 7)})
    ) == 2


def test_relation_connectivity_reports_chartwise_gap() -> None:
    grid = Grid(bounds=[[0.0, 1.0]], subdivisions=[2])
    incidence = AugmentedQuotientIncidence(grid, _ingredients())
    relation = {
        BaseCell(0): frozenset({BaseCell(0), PhaseCell(1, 15)}),
        BaseCell(1): frozenset({BaseCell(0), PhaseCell(1, 7)}),
    }

    audit = audit_relation_image_connectivity(relation, incidence)

    assert not audit.passed
    assert len(audit.disconnected) == 1
    assert audit.disconnected[0].label == BaseCell(1)
    assert len(audit.disconnected[0].components) == 2
    with pytest.raises(RelationAuditFailure, match="disconnected image values"):
        audit.require_passed(context="quotient benchmark")
