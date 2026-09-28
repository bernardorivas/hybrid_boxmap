"""
Impacting van der Pol-Duffing oscillator

A particle in the double-well potential ``V(x) = x^4/4 - x^2/2`` with van der
Pol damping ``eps (x^2 - beta)`` (negative for ``|x| < sqrt(beta)``) hits a
rigid stop at ``x = w`` inside the right well and rebounds with coefficient of
restitution ``c``.

System:
- State: [x, v] (position, velocity), with x <= w
- Continuous dynamics: x' = v, v' = x - x^3 + eps (beta - x^2) v
- Jump condition: x = w and v >= 0 (the particle at the stop, moving into it)
- Reset map: [w, v] -> [w, -c v]

The default parameters are ``eps = 1``, ``beta = 0.8``, ``w = 0.8``,
``c = 0.7`` and the window ``R = [-1.95, 0.8] x [-2.35, 1.95]``.  Numerically
(pre-impact speeds ``u``, suspension periods with unit handles), the
invariant sets in ``R`` are

* ``F = (-1, 0)``, a stable focus (eigenvalues ``-0.1 +- 1.4107 i``);
* ``Z = (w, 0)``, a Zeno point (``x - x^3 = 0.288 > 0`` at the stop);
* ``C``, an attracting impact cycle with one impact per period
  (``u = 1.6905562``, multiplier ``0.24477``, period ``5.4961``,
  ``x`` in ``[-1.7429, 0.8]``);
* ``S = (0, 0)``, a saddle (eigenvalues ``1.47703`` and ``-0.67703``);
* ``U_Z``, a repelling impact cycle around ``Z`` with one impact per period
  (``u = 0.6208138``, multiplier ``2.0245``, period ``3.9170``, ``x`` in
  ``[0.4236, 0.8]``);

with Hasse diagram ``U_Z -> Z``, ``U_Z -> S``, ``S -> F``, ``S -> C``.
With the other parameters at these values the structure persists for
``beta`` in ``(0.7147, 0.8138)``: at ``0.8138`` the left branches of the
stable and unstable manifolds of ``S`` form a homoclinic loop around ``F``,
and below ``0.7147`` the right unstable branch lands, after one impact, in the
basin of ``F``.

The Lienard function ``L = (v + eps Gf(x))^2 / 2 + V(x)`` with
``Gf(x) = int_w^x (s^2 - beta) ds`` satisfies ``dL/dt = -eps Gf(x) V'(x)``
along the flow; since ``Gf(w) = 0``, ``L`` equals the energy at the stop and
each impact with speed ``v`` lowers it by ``(1 - c^2) v^2 / 2``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..src.batched_suspension_flow import BatchDynamics
from ..src.hybrid_system import HybridSystem


def potential(x):
    """Double-well potential ``V(x) = x^4/4 - x^2/2``."""

    return 0.25 * x**4 - 0.5 * x**2


def potential_derivative(x):
    """``V'(x) = x^3 - x``."""

    return x**3 - x


def energy(state):
    """Mechanical energy ``E = v^2/2 + V(x)`` of states ``[..., (x, v)]``."""

    state = np.asarray(state, dtype=float)
    return 0.5 * state[..., 1] ** 2 + potential(state[..., 0])


def ode_fun(t: float, state: np.ndarray, oscillator) -> np.ndarray:
    """
    Continuous dynamics: x' = v, v' = x - x^3 + eps (beta - x^2) v

    Args:
        t: Time (unused, autonomous system)
        state: [position, velocity]
        oscillator: ImpactVanDerPolDuffing instance with parameters

    Returns:
        Time derivatives [dx/dt, dv/dt]
    """
    x, v = state
    return np.array([v, x - x**3 + oscillator.eps * (oscillator.beta - x * x) * v])


def event_fun(t: float, state: np.ndarray, oscillator) -> float:
    """
    Event function for impacts with the stop.

    Returns ``x - w`` while the particle moves toward the stop (``v >= 0``)
    and ``-1`` otherwise, so that a post-impact state ``(w, -c v)`` is not an
    event point (the mirror image of the bouncing-ball convention).  With
    ``direction = +1`` the guard is ``{(w, v) : v >= 0}``.

    Args:
        t: Time (unused)
        state: [position, velocity]
        oscillator: ImpactVanDerPolDuffing instance with parameters

    Returns:
        Signed distance to the stop (zero at an impact)
    """
    x, v = state
    return x - oscillator.w if v >= 0 else -1.0


def reset_map(state: np.ndarray, oscillator) -> np.ndarray:
    """
    Reset map for impacts: [w, v] -> [w, -c v].

    Args:
        state: [position, velocity] at impact
        oscillator: ImpactVanDerPolDuffing instance with parameters

    Returns:
        Post-impact state
    """
    _, v = state
    return np.array([oscillator.w, -oscillator.c * v])


class ImpactVanDerPolDuffing:
    """Van der Pol-Duffing oscillator with a rigid stop at ``x = w``."""

    def __init__(
        self,
        domain_bounds: list[tuple[float, float]] | None = None,
        eps: float = 1.0,
        beta: float = 0.8,
        w: float = 0.8,
        c: float = 0.7,
        max_jumps: int = 200,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ):
        if not 0.0 < w < 1.0:
            raise ValueError("the stop w must lie in (0, 1), so that x - x^3 > 0 there")
        if not 0.0 < c < 1.0:
            raise ValueError("the coefficient of restitution c must lie in (0, 1)")
        self.eps = float(eps)
        self.beta = float(beta)
        self.w = float(w)
        self.c = float(c)
        # The window R of the manuscript example.
        self.domain_bounds = (
            [(-1.95, self.w), (-2.35, 1.95)] if domain_bounds is None else domain_bounds
        )
        self.rtol = rtol
        self.atol = atol
        self.max_jumps = max_jumps

        def event(t: float, state: np.ndarray) -> float:
            return event_fun(t, state, self)

        event.terminal = True
        event.direction = 1  # x - w increases through zero at an impact
        self.event_function = event

        self.system = self._create_system()

    def _create_system(self) -> HybridSystem:
        """Create the hybrid system with closures capturing self."""

        def ode(t: float, state: np.ndarray) -> np.ndarray:
            return ode_fun(t, state, self)

        def reset(state: np.ndarray) -> np.ndarray:
            return reset_map(state, self)

        return HybridSystem(
            ode=ode,
            event_function=self.event_function,
            reset_map=reset,
            domain_bounds=self.domain_bounds,
            max_jumps=self.max_jumps,
            event_direction=1,
            rtol=self.rtol,
            atol=self.atol,
        )

    def batch_dynamics(self) -> BatchDynamics:
        """:func:`ode_fun`, :func:`event_fun`, and :func:`reset_map` on arrays of states."""

        eps = self.eps
        beta = self.beta
        w = self.w
        c = self.c

        def vector_field(states: np.ndarray) -> np.ndarray:
            x = states[:, 0]
            v = states[:, 1]
            return np.stack((v, x - x**3 + eps * (beta - x * x) * v), axis=1)

        def event(states: np.ndarray) -> np.ndarray:
            return np.where(states[:, 1] >= 0, states[:, 0] - w, -1.0)

        def reset(states: np.ndarray) -> np.ndarray:
            v = states[:, 1]
            return np.stack((np.full_like(v, w), -c * v), axis=1)

        return BatchDynamics(vector_field=vector_field, event=event, reset=reset)

    def simulate(self, initial_state: np.ndarray, time_span: tuple[float, float]):
        """Simulate the oscillator."""
        return self.system.simulate(initial_state, time_span, dense_output=True)

    def equilibria(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The equilibria ``F = (-1, 0)`` and ``S = (0, 0)`` in ``x <= w``, with
        the eigenvalues of the Jacobian."""

        result = {}
        for name, x in (("F", -1.0), ("S", 0.0)):
            jacobian = np.array(
                [[0.0, 1.0], [1.0 - 3.0 * x * x, self.eps * (self.beta - x * x)]]
            )
            result[name] = (np.array([x, 0.0]), np.linalg.eigvals(jacobian))
        return result

    def lienard_primitive(self, x):
        """``Gf(x) = int_w^x (s^2 - beta) ds``, so that ``Gf(w) = 0``."""

        return (x**3 - self.w**3) / 3.0 - self.beta * (x - self.w)

    def lienard_function(self, state):
        """``L = (v + eps Gf(x))^2 / 2 + V(x)`` of states ``[..., (x, v)]``."""

        state = np.asarray(state, dtype=float)
        x, v = state[..., 0], state[..., 1]
        y = v + self.eps * self.lienard_primitive(x)
        return 0.5 * y * y + potential(x)

    def lienard_rate(self, x):
        """``dL/dt = -eps Gf(x) V'(x)`` along the flow; it depends only on ``x``."""

        return -self.eps * self.lienard_primitive(x) * potential_derivative(x)

    def impact_loss(self, v):
        """Decrease ``(1 - c^2) v^2 / 2`` of ``E`` and ``L`` at an impact with speed ``v``."""

        return 0.5 * (1.0 - self.c**2) * np.asarray(v, dtype=float) ** 2

    def impact_return(
        self, speed: float, *, t_max: float = 100.0, max_step: float | None = 0.02
    ) -> tuple[float, float, object]:
        """Flight from the reset of an impact with speed ``speed`` to the next impact.

        Returns ``(next pre-impact speed, flight time, dense solution)``; the
        speed is ``nan`` if there is no impact before ``t_max``.
        """

        options = {} if max_step is None else {"max_step": max_step}
        solution = solve_ivp(
            self.system.ode,
            (0.0, float(t_max)),
            reset_map(np.array([self.w, float(speed)]), self),
            events=self.event_function,
            dense_output=True,
            rtol=self.rtol,
            atol=self.atol,
            **options,
        )
        if solution.t_events[0].size == 0:
            return float("nan"), float("inf"), solution.sol
        return (
            float(solution.y_events[0][0][1]),
            float(solution.t_events[0][0]),
            solution.sol,
        )

    def impact_cycle(self, bracket: tuple[float, float]) -> float:
        """Pre-impact speed of the one-impact cycle whose speed lies in ``bracket``."""

        return float(
            brentq(lambda u: self.impact_return(u)[0] - u, *bracket, xtol=1.0e-12)
        )
