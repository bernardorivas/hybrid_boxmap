"""The examples of the manuscript on the paper suspension grid.

Each factory returns a :class:`SuspensionGridProblem` with the system, the
base window ``R`` (as dyadic grids ``X_j`` with ``2**j`` cells per axis of the
ambient rectangle), the guard parametrization of ``G cap R``, the reset on it,
and the time ``tau`` stated in the Examples section:

* bouncing ball: ``g = 9.81``, ``c = 0.8``, ``R = [0,2] x [-5,5]``,
  ``G cap R = {(0, v) : -5 <= v <= 0}``, ``r(0, v) = (0, -c v)``,
  ``tau = 1.5``;
* rimless wheel: ``alpha = 0.4``, ``gamma = 0.2``,
  ``R = [-0.2,0.6] x [-0.5,1]``,
  ``G cap R = {(alpha+gamma, omega) : 0 <= omega <= 1}``,
  ``r(alpha+gamma, omega) = (gamma-alpha, cos(2 alpha) omega)``, ``tau = 2``;
* spiking neuron: the L-shaped ``X = R``, ambient rectangle
  ``[-120,200] x [-400,880]``, ``G = {(35, u) : -300 <= u <= 160}``,
  ``r(35, u) = (-50, u + 100)``, ``tau = 20``;
* impacting van der Pol-Duffing oscillator: ``eps = 1``, ``beta = 0.8``,
  ``w = 0.8``, ``c = 0.7``, ``R = [-1.95,w] x [-2.35,1.95]``,
  ``G cap R = {(w, v) : 0 <= v <= 1.95}``, ``r(w, v) = (w, -c v)``,
  ``tau = 3`` by default (the runs also use other values of ``tau``).

The factories are module-level functions so that worker processes can rebuild
a problem from a :func:`functools.partial`.
"""

from __future__ import annotations

import functools

import numpy as np

from ..src.suspension_grid import DyadicBaseWindow, GuardResetSpec
from ..src.suspension_grid_relation import SuspensionGridProblem
from .bouncing_ball import BouncingBall
from .impact_vdp_duffing import ImpactVanDerPolDuffing
from .rimless_wheel import RimlessWheel
from .spiking_neuron import (
    U_MAX,
    U_MIN,
    U_NOTCH,
    U_RESET_SHIFT,
    V_MIN,
    V_NOTCH,
    V_PEAK,
    V_RESET,
    SpikingNeuron,
)


BALL_BOUNDS = ((0.0, 2.0), (-5.0, 5.0))
WHEEL_BOUNDS = ((-0.2, 0.6), (-0.5, 1.0))
NEURON_AMBIENT_BOUNDS = ((-120.0, 200.0), (-400.0, 880.0))
#: ``R = [-1.95, w] x [-2.35, 1.95]`` for the stop ``w = 0.8``.
IMPACT_X_MIN = -1.95
IMPACT_V_BOUNDS = (-2.35, 1.95)


def bouncing_ball_problem(
    *,
    gravity: float = 9.81,
    restitution: float = 0.8,
    tau: float = 1.5,
    max_step: float | None = 0.02,
) -> SuspensionGridProblem:
    ball = BouncingBall(
        domain_bounds=[(-1.0e6, 1.0e6), (-1.0e6, 1.0e6)],
        g=float(gravity),
        c=float(restitution),
    )
    c = float(restitution)
    guard = GuardResetSpec(
        u_bounds=(BALL_BOUNDS[1][0], 0.0),
        guard_point=lambda u: np.stack((np.zeros_like(u), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.zeros_like(u), -c * u), axis=-1),
        name="ball guard h=0, v<=0; reset v -> -c v",
    )
    window = DyadicBaseWindow(BALL_BOUNDS, name="R=[0,2]x[-5,5]")
    return SuspensionGridProblem(
        system=ball.system,
        window=window,
        guard=guard,
        tau=float(tau),
        max_step=max_step,
        name="bouncing-ball",
        parameters={
            "g": float(gravity),
            "c": c,
            "R": [list(interval) for interval in BALL_BOUNDS],
            "tau": float(tau),
            "max_step": max_step,
            "rtol": ball.rtol,
            "atol": ball.atol,
        },
    )


def rimless_wheel_problem(
    *,
    alpha: float = 0.4,
    gamma: float = 0.2,
    tau: float = 2.0,
    max_step: float | None = 0.02,
) -> SuspensionGridProblem:
    wheel = RimlessWheel(
        domain_bounds=[(-1.0e6, 1.0e6), (-1.0e6, 1.0e6)],
        alpha=float(alpha),
        gamma=float(gamma),
    )
    guard_angle = float(alpha + gamma)
    reset_angle = float(gamma - alpha)
    factor = float(np.cos(2.0 * alpha))
    guard = GuardResetSpec(
        u_bounds=(0.0, WHEEL_BOUNDS[1][1]),
        guard_point=lambda u: np.stack((np.full_like(u, guard_angle), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.full_like(u, reset_angle), factor * u), axis=-1),
        name="wheel guard theta=alpha+gamma, omega>=0; reset (gamma-alpha, cos(2 alpha) omega)",
    )
    window = DyadicBaseWindow(WHEEL_BOUNDS, name="R=[-0.2,0.6]x[-0.5,1]")
    return SuspensionGridProblem(
        system=wheel.system,
        window=window,
        guard=guard,
        tau=float(tau),
        max_step=max_step,
        name="rimless-wheel",
        parameters={
            "alpha": float(alpha),
            "gamma": float(gamma),
            "R": [list(interval) for interval in WHEEL_BOUNDS],
            "tau": float(tau),
            "max_step": max_step,
            "rtol": wheel.rtol,
            "atol": wheel.atol,
        },
    )


def _neuron_region(centers: np.ndarray) -> np.ndarray:
    v = centers[:, 0]
    u = centers[:, 1]
    left = (v > V_MIN) & (v < V_NOTCH) & (u > U_MIN) & (u < U_MAX)
    right = (v > V_NOTCH) & (v < V_PEAK) & (u > U_MIN) & (u < U_NOTCH)
    return left | right


def spiking_neuron_problem(
    *,
    tau: float = 20.0,
    max_step: float | None = 0.02,
) -> SuspensionGridProblem:
    neuron = SpikingNeuron(max_jumps=1000)
    guard = GuardResetSpec(
        u_bounds=(U_MIN, U_NOTCH),
        guard_point=lambda u: np.stack((np.full_like(u, V_PEAK), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.full_like(u, V_RESET), u + U_RESET_SHIFT), axis=-1),
        name="neuron guard v=35; reset (v,u) -> (-50, u+100)",
    )
    window = DyadicBaseWindow(
        NEURON_AMBIENT_BOUNDS,
        region=_neuron_region,
        name="L-shaped X=([-80,-40]x[-300,600]) u ([-40,35]x[-300,160]) in [-120,200]x[-400,880]",
    )
    return SuspensionGridProblem(
        system=neuron.system,
        window=window,
        guard=guard,
        tau=float(tau),
        max_step=max_step,
        name="spiking-neuron",
        parameters={
            "X": "([-80,-40]x[-300,600]) u ([-40,35]x[-300,160])",
            "ambient": [list(interval) for interval in NEURON_AMBIENT_BOUNDS],
            "tau": float(tau),
            "max_step": max_step,
            "rtol": neuron.rtol,
            "atol": neuron.atol,
        },
    )


def impact_vdp_duffing_problem(
    *,
    eps: float = 1.0,
    beta: float = 0.8,
    stop: float = 0.8,
    restitution: float = 0.7,
    tau: float = 3.0,
    max_step: float | None = 0.02,
) -> SuspensionGridProblem:
    oscillator = ImpactVanDerPolDuffing(
        domain_bounds=[(-1.0e6, 1.0e6), (-1.0e6, 1.0e6)],
        eps=float(eps),
        beta=float(beta),
        w=float(stop),
        c=float(restitution),
    )
    w = oscillator.w
    c = oscillator.c
    bounds = ((IMPACT_X_MIN, w), IMPACT_V_BOUNDS)
    guard = GuardResetSpec(
        u_bounds=(0.0, IMPACT_V_BOUNDS[1]),
        guard_point=lambda u: np.stack((np.full_like(u, w), u), axis=-1),
        guard_coordinate=lambda x: x[:, 1],
        reset_point=lambda u: np.stack((np.full_like(u, w), -c * u), axis=-1),
        name="impact guard x=w, v>=0; reset v -> -c v",
    )
    window = DyadicBaseWindow(bounds, name=f"R=[{IMPACT_X_MIN},{w}]x[-2.35,1.95]")
    return SuspensionGridProblem(
        system=oscillator.system,
        window=window,
        guard=guard,
        tau=float(tau),
        max_step=max_step,
        name="impact-vdp-duffing",
        parameters={
            "eps": oscillator.eps,
            "beta": oscillator.beta,
            "w": w,
            "c": c,
            "R": [list(interval) for interval in bounds],
            "tau": float(tau),
            "max_step": max_step,
            "rtol": oscillator.rtol,
            "atol": oscillator.atol,
        },
    )


def impact_vdp_duffing_reference_sets(
    problem: SuspensionGridProblem, *, samples: int = 400
) -> dict[str, dict[str, np.ndarray]]:
    """Points on the invariant sets ``F``, ``Z``, ``C``, ``S``, ``U_Z``.

    Each set is given by base points ``"base"`` (rows ``(x, v)``) and handle
    points ``"handle"`` (rows ``(u, s)``).  The impact cycles are the fixed
    points of the impact map with pre-impact speed in ``[1.5, 1.9]`` (``C``)
    and ``[0.55, 0.66]`` (``U_Z``), where they lie for the default
    parameters; a cycle that is not bracketed there is omitted.  Used to
    identify computed Morse sets.
    """

    parameters = problem.parameters
    oscillator = ImpactVanDerPolDuffing(
        domain_bounds=[(-1.0e6, 1.0e6), (-1.0e6, 1.0e6)],
        eps=parameters["eps"],
        beta=parameters["beta"],
        w=parameters["w"],
        c=parameters["c"],
    )
    w = oscillator.w
    empty = np.zeros((0, 2))
    phases = (np.arange(samples) + 0.5) / samples
    sets: dict[str, dict[str, np.ndarray]] = {
        "F": {"base": np.array([[-1.0, 0.0]]), "handle": empty},
        "S": {"base": np.array([[0.0, 0.0]]), "handle": empty},
        "Z": {"base": np.array([[w, 0.0]]), "handle": np.stack((np.zeros(samples), phases), axis=1)},
    }
    for name, bracket in (("C", (1.5, 1.9)), ("U_Z", (0.55, 0.66))):
        try:
            speed = oscillator.impact_cycle(bracket)
        except ValueError:
            continue
        _speed, flight, solution = oscillator.impact_return(speed)
        times = np.linspace(0.0, flight, samples)
        sets[name] = {
            "base": np.asarray(solution(times)).T,
            "handle": np.stack((np.full(samples, speed), phases), axis=1),
        }
    return sets


#: Named invariant sets used to identify Morse sets in the runner.
PAPER_GRID_REFERENCE_SETS = {
    "impact-vdp-duffing": impact_vdp_duffing_reference_sets,
}


PAPER_GRID_PROBLEMS = {
    "bouncing-ball": bouncing_ball_problem,
    "rimless-wheel": rimless_wheel_problem,
    "spiking-neuron": spiking_neuron_problem,
    "impact-vdp-duffing": impact_vdp_duffing_problem,
}


def paper_grid_problem_factory(name: str, **kwargs: object) -> functools.partial:
    """A picklable zero-argument factory for worker processes."""

    return functools.partial(PAPER_GRID_PROBLEMS[name], **kwargs)


__all__ = [
    "BALL_BOUNDS",
    "IMPACT_V_BOUNDS",
    "IMPACT_X_MIN",
    "NEURON_AMBIENT_BOUNDS",
    "PAPER_GRID_PROBLEMS",
    "PAPER_GRID_REFERENCE_SETS",
    "WHEEL_BOUNDS",
    "bouncing_ball_problem",
    "impact_vdp_duffing_problem",
    "impact_vdp_duffing_reference_sets",
    "paper_grid_problem_factory",
    "rimless_wheel_problem",
    "spiking_neuron_problem",
]
