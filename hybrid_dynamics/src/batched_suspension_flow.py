"""Endpoints of the suspension semiflow for many initial points at once.

:class:`SuspensionFlow` integrates one path at a time with
:func:`scipy.integrate.solve_ivp` (``method="RK45"``), and most of the time
of a relation is spent in the per-call overhead of that solver.  This module
runs the same method on an array of points: every point follows the step
sequence that ``solve_ivp`` would give it, so the endpoints agree with those
of :class:`SuspensionFlow` up to rounding.  Per point it reproduces

* the Dormand--Prince 5(4) pair of ``scipy.integrate.RK45``, its initial step
  (``select_initial_step``), its error norm, its step factors (safety
  ``0.9``, factors in ``[0.2, 10]``, no increase right after a rejected
  step), ``max_step``, and the clipping of the last step to the end of the
  time span;
* the terminal event with a crossing direction, detected as in
  ``find_active_events`` from the values of the event function at the ends
  of each accepted step, and located on the fourth-order dense output of the
  step (bisection instead of ``brentq``, with ``brentq``'s rule that a zero
  at the left end of the step is the root);
* the unit-time handles of the suspension: at an event the path spends one
  unit of time on the handle of the guard point and continues from the
  reset;
* the window-exit records of :class:`SuspensionPath`: the initial point of
  each flow segment, the end of each accepted step, and the event point are
  tested with ``in_window``, and the guard point of each handle with
  ``guard_in_window``.

A point whose integration ``solve_ivp`` would stop with an error (a step
below the minimal step, a value that is not finite) or that needs more
steps than the cap is reported as unresolved; the caller evaluates it with
:class:`SuspensionFlow`, so its result, including a failure, is exactly the
reference one.

The vector field, event function, and reset are given in vectorized form by
:class:`BatchDynamics`; each example supplies one whose arithmetic is that of
its scalar functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

#: Endpoint kinds, the codes of :mod:`suspension_grid_relation`.
KIND_FAILED = -1
KIND_BASE = 0
KIND_HANDLE = 1

# Dormand--Prince 5(4), as in scipy.integrate._ivp.rk.RK45.
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0])
_A = (
    np.array([]),
    np.array([1 / 5]),
    np.array([3 / 40, 9 / 40]),
    np.array([44 / 45, -56 / 15, 32 / 9]),
    np.array([19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]),
    np.array([9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]),
)
_B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
_E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40])
_P = np.array(
    [
        [1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
        [0, 0, 0, 0],
        [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
        [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
        [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
        [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
    ]
)
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_ERROR_EXPONENT = -1.0 / 5.0
_ORDER = 4
_BISECTIONS = 60


@dataclass(frozen=True)
class BatchDynamics:
    """Vectorized vector field, event function, and reset of a planar system.

    Each function maps states of shape ``(n, 2)`` to the values at every row:
    ``vector_field`` and ``reset`` to arrays of shape ``(n, 2)`` and ``event``
    to an array of shape ``(n,)``.  They must compute what the scalar
    functions of the :class:`HybridSystem` compute, with the same arithmetic,
    since the batched endpoints are compared with the scalar ones.  The
    crossing direction and the tolerances are read from the system.
    """

    vector_field: Callable[[FloatArray], FloatArray]
    event: Callable[[FloatArray], FloatArray]
    reset: Callable[[FloatArray], FloatArray]


@dataclass
class BatchedEndpoints:
    """Endpoints at time ``tau``; rows with ``unresolved`` are not filled."""

    kind: npt.NDArray[np.int64]
    state: FloatArray
    phase: FloatArray
    left_window: BoolArray
    unresolved: BoolArray


def _rms(values: FloatArray) -> FloatArray:
    return np.sqrt(np.mean(values * values, axis=1))


def _initial_step(
    field: Callable[[FloatArray], FloatArray],
    y0: FloatArray,
    f0: FloatArray,
    t_bound: FloatArray,
    max_step: float,
    rtol: float,
    atol: float,
) -> FloatArray:
    """``scipy.integrate._ivp.common.select_initial_step`` for every row."""

    scale = atol + np.abs(y0) * rtol
    d0 = _rms(y0 / scale)
    d1 = _rms(f0 / scale)
    small = (d0 < 1e-5) | (d1 < 1e-5)
    h0 = np.where(small, 1e-6, 0.01 * d0 / np.where(small, 1.0, d1))
    h0 = np.minimum(h0, t_bound)
    f1 = field(y0 + h0[:, None] * f0)
    d2 = _rms((f1 - f0) / scale) / h0
    flat = (d1 <= 1e-15) & (d2 <= 1e-15)
    largest = np.maximum(d1, d2)
    h1 = np.where(
        flat,
        np.maximum(1e-6, h0 * 1e-3),
        (0.01 / np.where(flat, 1.0, largest)) ** (1.0 / (_ORDER + 1)),
    )
    return np.minimum(np.minimum(np.minimum(100 * h0, h1), t_bound), max_step)


def _dense_matrices(stages: FloatArray) -> FloatArray:
    """``Q = K^T P`` of the dense output of each step, shape ``(n, 2, 4)``."""

    return np.einsum("nsd,sk->ndk", stages, _P)


def _dense_value(y_old: FloatArray, h: FloatArray, q: FloatArray, x: FloatArray) -> FloatArray:
    """``y_old + h Q [x, x^2, x^3, x^4]``, as ``RkDenseOutput`` evaluates it."""

    powers = np.cumprod(np.repeat(x[:, None], _ORDER, axis=1), axis=1)
    return h[:, None] * np.einsum("ndk,nk->nd", q, powers) + y_old


def batched_suspension_endpoints(
    dynamics: BatchDynamics,
    points: FloatArray,
    start_on_handle: BoolArray,
    tau: float,
    *,
    direction: float,
    rtol: float,
    atol: float,
    max_step: float | None,
    in_window: Callable[[FloatArray], BoolArray] | None = None,
    guard_in_window: Callable[[FloatArray], BoolArray] | None = None,
    path_atol: float = 1.0e-9,
    max_iterations: int = 1_000_000,
) -> BatchedEndpoints:
    """Endpoints at suspension time ``tau`` of the paths from ``points``.

    A point with ``start_on_handle`` starts on the handle of the guard point
    it is, at phase zero.  The result has, per point, the kind of endpoint
    (base point, or a handle point given by its guard point and phase), the
    state, the phase, and whether the path was recorded outside the window
    before time ``tau - path_atol`` (the ``exit_time`` rule of
    :class:`SuspensionPath`).
    """

    field = dynamics.vector_field
    event = dynamics.event
    reset = dynamics.reset
    tau = float(tau)
    max_step = np.inf if max_step is None else float(max_step)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = points.shape[0]
    kind = np.full(n, KIND_FAILED, dtype=np.int64)
    state = np.full((n, 2), np.nan)
    phase = np.full(n, np.nan)
    left = np.zeros(n, dtype=bool)
    unresolved = np.zeros(n, dtype=bool)

    y = points.copy()
    clock = np.zeros(n)  # suspension time at the start of the current segment
    t_local = np.zeros(n)  # time within the current flow segment
    t_bound = np.zeros(n)
    h_abs = np.zeros(n)
    f_y = np.zeros((n, 2))
    g_y = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    flowing = np.zeros(n, dtype=bool)

    def outside_window(values: FloatArray) -> BoolArray:
        if in_window is None:
            return np.zeros(values.shape[0], dtype=bool)
        return ~np.asarray(in_window(values), dtype=bool)

    def start_flows(rows: npt.NDArray[np.int64]) -> None:
        remaining = tau - clock[rows]
        stop = remaining <= path_atol
        done = rows[stop]
        kind[done] = KIND_BASE
        state[done] = y[done]
        rows = rows[~stop]
        if rows.size == 0:
            return
        left[rows[outside_window(y[rows])]] = True
        t_local[rows] = 0.0
        t_bound[rows] = tau - clock[rows]
        f_y[rows] = field(y[rows])
        g_y[rows] = event(y[rows])
        h_abs[rows] = _initial_step(field, y[rows], f_y[rows], t_bound[rows], max_step, rtol, atol)
        rejected[rows] = False
        flowing[rows] = True

    def start_handles(rows: npt.NDArray[np.int64]) -> None:
        # The handle of the guard point y[rows] starts at suspension time clock[rows].
        if guard_in_window is not None:
            outside = ~np.asarray(guard_in_window(y[rows]), dtype=bool)
            left[rows[outside & (clock[rows] < tau - path_atol)]] = True
        end = clock[rows] + 1.0
        at_start = np.abs(tau - clock[rows]) <= path_atol
        at_end = ~at_start & (np.abs(tau - end) <= path_atol)
        inside = ~at_start & ~at_end & (tau > clock[rows]) & (tau < end)
        kind[rows[at_start]] = KIND_BASE
        state[rows[at_start]] = y[rows[at_start]]
        kind[rows[inside]] = KIND_HANDLE
        state[rows[inside]] = y[rows[inside]]
        phase[rows[inside]] = tau - clock[rows[inside]]
        passing = rows[~at_start & ~inside]
        if passing.size:
            y[passing] = reset(y[passing])
        kind[rows[at_end]] = KIND_BASE
        state[rows[at_end]] = y[rows[at_end]]
        onward = ~at_start & ~at_end & ~inside
        clock[rows[onward]] = end[onward]
        start_flows(rows[onward])

    start_flows(np.flatnonzero(~start_on_handle))
    start_handles(np.flatnonzero(start_on_handle))

    for _ in range(int(max_iterations)):
        rows = np.flatnonzero(flowing)
        if rows.size == 0:
            break
        t0 = t_local[rows]
        y0 = y[rows]
        f0 = f_y[rows]
        bound = t_bound[rows]
        step = h_abs[rows]
        # At the start of a step scipy clips h_abs to [min_step, max_step]; a
        # retried step below min_step is a solver failure.
        min_step = 10.0 * np.abs(np.nextafter(t0, np.inf) - t0)
        fresh = ~rejected[rows]
        step = np.where(fresh, np.clip(step, min_step, max_step), step)
        too_small = ~fresh & (step < min_step)
        if np.any(too_small):
            unresolved[rows[too_small]] = True
            flowing[rows[too_small]] = False
            keep = ~too_small
            rows, t0, y0, f0, bound, step = (
                rows[keep], t0[keep], y0[keep], f0[keep], bound[keep], step[keep]
            )
            if rows.size == 0:
                continue
        t_new = t0 + step
        t_new = np.where(t_new - bound > 0, bound, t_new)
        h = t_new - t0
        stages = np.empty((rows.size, 7, 2))
        stages[:, 0] = f0
        for s in range(1, 6):
            dy = (stages[:, :s].transpose(0, 2, 1) @ _A[s]) * h[:, None]
            stages[:, s] = field(y0 + dy)
        y_new = y0 + h[:, None] * (stages[:, :6].transpose(0, 2, 1) @ _B)
        f_new = field(y_new)
        stages[:, 6] = f_new
        scale = atol + np.maximum(np.abs(y0), np.abs(y_new)) * rtol
        error = _rms(((stages.transpose(0, 2, 1) @ _E) * h[:, None]) / scale)
        broken = ~np.isfinite(error) | ~np.all(np.isfinite(y_new), axis=1)
        if np.any(broken):
            unresolved[rows[broken]] = True
            flowing[rows[broken]] = False
        accept = (error < 1) & ~broken
        retry = ~accept & ~broken
        positive = np.where(error > 0, error, 1.0)
        grow = np.where(error == 0, _MAX_FACTOR, np.minimum(_MAX_FACTOR, _SAFETY * positive**_ERROR_EXPONENT))
        grow = np.where(rejected[rows], np.minimum(1.0, grow), grow)
        shrink = np.maximum(_MIN_FACTOR, _SAFETY * positive**_ERROR_EXPONENT)
        h_abs[rows[accept]] = np.abs(h[accept]) * grow[accept]
        h_abs[rows[retry]] = np.abs(h[retry]) * shrink[retry]
        rejected[rows[retry]] = True
        if not np.any(accept):
            continue
        ia = np.flatnonzero(accept)
        done_rows = rows[ia]
        rejected[done_rows] = False
        g_new = event(y_new[ia])
        g_old = g_y[done_rows]
        if direction > 0:
            crossed = (g_old <= 0) & (g_new >= 0)
        elif direction < 0:
            crossed = (g_old >= 0) & (g_new <= 0)
        else:
            crossed = ((g_old <= 0) & (g_new >= 0)) | ((g_old >= 0) & (g_new <= 0))
        if np.any(crossed):
            ic = ia[crossed]
            event_rows = rows[ic]
            q = _dense_matrices(stages[ic])
            lower = np.zeros(ic.size)
            upper = np.ones(ic.size)
            for _bisection in range(_BISECTIONS):
                middle = 0.5 * (lower + upper)
                value = event(_dense_value(y0[ic], h[ic], q, middle))
                if direction > 0:
                    hit = value >= 0
                elif direction < 0:
                    hit = value <= 0
                else:
                    start_sign = np.sign(g_old[crossed])
                    hit = np.sign(value) != start_sign
                upper = np.where(hit, middle, upper)
                lower = np.where(hit, lower, middle)
            # brentq returns the left end of the step when the event function
            # vanishes there: a path that starts on the event surface outside
            # the guard and crosses it within its first step has its event at
            # time zero, as in solve_ivp.
            upper = np.where(g_old[crossed] == 0, 0.0, upper)
            y_event = _dense_value(y0[ic], h[ic], q, upper)
            t_event = t0[ic] + upper * h[ic]
            late = clock[event_rows] + t_event < tau - path_atol
            left[event_rows[outside_window(y_event) & late]] = True
            clock[event_rows] = clock[event_rows] + t_event
            y[event_rows] = y_event
            flowing[event_rows] = False
            start_handles(event_rows)
        ip = ia[~crossed]
        plain = rows[ip]
        late = clock[plain] + t_new[ip] < tau - path_atol
        left[plain[outside_window(y_new[ip]) & late]] = True
        finished = t_new[ip] == bound[ip]
        moving = plain[~finished]
        t_local[moving] = t_new[ip][~finished]
        y[moving] = y_new[ip][~finished]
        f_y[moving] = f_new[ip][~finished]
        g_y[moving] = g_new[~crossed][~finished]
        # The endpoint at the end of the time span is read from the dense
        # output of the last step at x = 1, as SuspensionPath.evaluate does.
        ending = ip[finished]
        if ending.size:
            q = _dense_matrices(stages[ending])
            end_rows = rows[ending]
            kind[end_rows] = KIND_BASE
            state[end_rows] = _dense_value(y0[ending], h[ending], q, np.ones(ending.size))
            y[end_rows] = y_new[ending]
            flowing[end_rows] = False
    else:
        unresolved[flowing] = True
    return BatchedEndpoints(kind=kind, state=state, phase=phase, left_window=left, unresolved=unresolved)


__all__ = [
    "BatchDynamics",
    "BatchedEndpoints",
    "batched_suspension_endpoints",
]
