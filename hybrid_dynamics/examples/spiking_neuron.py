"""Compact quadratic integrate-and-fire hybrid neuron.

The numerical benchmark is the parameter choice in
``literature-review/arXiv-2606.18501v1/examples/neuron.tex``.  Unlike the
paper's initial global description ``v <= v_peak``, this implementation uses
the compact L-shaped state space

``([-80,-40] x [-300,600]) union ([-40,35] x [-300,160])``.

That restriction is essential: the paper's compactness and transversality
hypotheses hold on this set, but not on the globally declared half-plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..src.hybrid_system import HybridSystem


V_MIN = -80.0
V_NOTCH = -40.0
V_RESET = -50.0
V_PEAK = 35.0
U_MIN = -300.0
U_NOTCH = 160.0
U_MAX = 600.0
U_RESET_SHIFT = 100.0


def in_spiking_neuron_domain(
    state: Sequence[float],
    *,
    atol: float = 1.0e-8,
) -> bool:
    """Return whether ``state`` lies in the closed compact L-shaped domain."""

    point = np.asarray(state, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return False
    v, u = (float(value) for value in point)
    in_bounding_rectangle = bool(
        V_MIN - atol <= v <= V_PEAK + atol
        and U_MIN - atol <= u <= U_MAX + atol
    )
    below_notch = u <= U_NOTCH + atol
    left_of_notch = v <= V_NOTCH + atol
    return in_bounding_rectangle and (below_notch or left_of_notch)


@dataclass(frozen=True)
class SpikingNeuronAnalyticAudit:
    """Exact extrema used to justify the compact computational restriction."""

    bottom_u_dot_min: float
    left_top_u_dot_max: float
    right_top_u_dot_max: float
    left_v_dot_min: float
    notch_v_dot_max: float
    guard_v_dot_min: float
    guard_v_dot_max: float
    reset_v_margin: float
    reset_u_lower_margin: float
    reset_u_upper_margin: float
    equilibrium_discriminant: float
    dwell_time_lower_bound: float

    @property
    def passed(self) -> bool:
        return bool(
            self.bottom_u_dot_min > 0.0
            and self.left_top_u_dot_max < 0.0
            and self.right_top_u_dot_max < 0.0
            and self.left_v_dot_min > 0.0
            and self.notch_v_dot_max < 0.0
            and self.guard_v_dot_min > 0.0
            and self.guard_v_dot_max >= self.guard_v_dot_min
            and self.reset_v_margin > 0.0
            and self.reset_u_lower_margin > 0.0
            and self.reset_u_upper_margin > 0.0
            and self.equilibrium_discriminant < 0.0
            and self.dwell_time_lower_bound > 0.0
        )

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "bottom_u_dot_min": self.bottom_u_dot_min,
            "left_top_u_dot_max": self.left_top_u_dot_max,
            "right_top_u_dot_max": self.right_top_u_dot_max,
            "left_v_dot_min": self.left_v_dot_min,
            "notch_v_dot_max": self.notch_v_dot_max,
            "guard_v_dot_min": self.guard_v_dot_min,
            "guard_v_dot_max": self.guard_v_dot_max,
            "reset_v_margin": self.reset_v_margin,
            "reset_u_lower_margin": self.reset_u_lower_margin,
            "reset_u_upper_margin": self.reset_u_upper_margin,
            "equilibrium_discriminant": self.equilibrium_discriminant,
            "dwell_time_lower_bound": self.dwell_time_lower_bound,
            "passed": self.passed,
        }


def spiking_neuron_analytic_audit() -> SpikingNeuronAnalyticAudit:
    """Return the exact boundary, reset, equilibrium, and dwell checks.

    The signs are extrema on the relevant straight boundary faces, not a
    numerical sampling of those faces.
    """

    guard_numerator_max = 5057.5 - U_MIN
    guard_numerator_min = 5057.5 - U_NOTCH
    return SpikingNeuronAnalyticAudit(
        bottom_u_dot_min=3.3,
        left_top_u_dot_max=-16.8,
        right_top_u_dot_max=-6.0,
        left_v_dot_min=0.3,
        notch_v_dot_max=-0.9,
        guard_v_dot_min=guard_numerator_min / 100.0,
        guard_v_dot_max=guard_numerator_max / 100.0,
        reset_v_margin=min(V_RESET - V_MIN, V_NOTCH - V_RESET),
        reset_u_lower_margin=(U_MIN + U_RESET_SHIFT) - U_MIN,
        reset_u_upper_margin=U_MAX - (U_NOTCH + U_RESET_SHIFT),
        equilibrium_discriminant=72.0**2 - 4.0 * 0.7 * 1870.0,
        dwell_time_lower_bound=(V_PEAK - V_RESET)
        / (guard_numerator_max / 100.0),
    )


class SpikingNeuron:
    """Quadratic integrate-and-fire model on its compact L-shaped domain."""

    def __init__(
        self,
        *,
        max_jumps: int = 20,
        rtol: float = 1.0e-10,
        atol: float = 1.0e-12,
    ) -> None:
        self.max_jumps = int(max_jumps)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.domain_bounds = [(V_MIN, V_PEAK), (U_MIN, U_MAX)]

        def ode(_time: float, state: np.ndarray) -> np.ndarray:
            v, u = np.asarray(state, dtype=np.float64)
            return np.asarray(
                (
                    (0.7 * (v + 60.0) * (v + 40.0) - u + 70.0) / 100.0,
                    0.03 * (-2.0 * (v + 60.0) - u),
                ),
                dtype=np.float64,
            )

        def event(_time: float, state: np.ndarray) -> float:
            return float(state[0] - V_PEAK)

        event.terminal = True
        event.direction = 1

        def reset(state: np.ndarray) -> np.ndarray:
            return np.asarray((V_RESET, float(state[1]) + U_RESET_SHIFT))

        self.system = HybridSystem(
            ode=ode,
            event_function=event,
            reset_map=reset,
            domain_bounds=self.domain_bounds,
            domain_predicate=in_spiking_neuron_domain,
            max_jumps=self.max_jumps,
            event_direction=1,
            rtol=self.rtol,
            atol=self.atol,
        )

    def simulate(
        self,
        initial_state: Sequence[float],
        time_span: tuple[float, float],
        *,
        max_jumps: int | None = None,
        max_step: float | None = None,
    ):
        return self.system.simulate(
            np.asarray(initial_state, dtype=np.float64),
            time_span,
            max_jumps=max_jumps,
            dense_output=True,
            max_step=max_step,
        )


__all__ = [
    "SpikingNeuron",
    "SpikingNeuronAnalyticAudit",
    "U_MAX",
    "U_MIN",
    "U_NOTCH",
    "U_RESET_SHIFT",
    "V_MIN",
    "V_NOTCH",
    "V_PEAK",
    "V_RESET",
    "in_spiking_neuron_domain",
    "spiking_neuron_analytic_audit",
]
