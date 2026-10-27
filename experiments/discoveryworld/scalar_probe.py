"""Optional public-feedback scalar probing, separate from learned decisions.

This module suggests targets; it executes no action and reads no environment
internals. A controller explicitly reports a failed point attempt before local
probes become eligible. All frequencies and uncertainties are expressed in Hz.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Iterable


_NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
_RANGE = re.compile(rf"\ballowable range is\s+({_NUMBER})\s+to\s+({_NUMBER})\s+(?:Hz|Hertz)\b", re.I)
_DELTA = re.compile(rf"^(increase|decrease)\s+.+?\s+by\s+({_NUMBER})\s+(?:Hz|Hertz)\s*$", re.I)
MAX_PROBE_ACTIONS = 16


def _decimal(value) -> Decimal:
    return Decimal(str(value).replace(",", ""))


@dataclass(frozen=True)
class PublicScalarControl:
    lower_hz: Decimal
    upper_hz: Decimal
    increments_hz: tuple[Decimal, ...]

    @property
    def minimum_step_hz(self) -> float:
        return float(min(abs(value) for value in self.increments_hz))


def public_scalar_control(dialog: dict) -> PublicScalarControl | None:
    """Parse the actual offered additive steps and range, never assume a grid."""
    if not dialog.get("is_in_dialog"):
        return None
    bounds = _RANGE.search(str(dialog.get("dialogIn", "")))
    if bounds is None:
        return None
    try:
        lower, upper = (_decimal(value) for value in bounds.group(1, 2))
        increments = set()
        for text in dialog.get("dialogOptions", {}).values():
            match = _DELTA.fullmatch(str(text).strip())
            if match is None:
                continue
            step = _decimal(match.group(2))
            if step > 0:
                increments.add(step if match.group(1).casefold() == "increase" else -step)
    except InvalidOperation:
        return None
    if not increments or lower >= upper:
        return None
    return PublicScalarControl(lower, upper, tuple(sorted(increments)))


def next_target(current: float, mean: float, std: float, public_dialog: dict,
                failed_values: Iterable[float], *, max_actions: int = MAX_PROBE_ACTIONS) -> float | None:
    """Choose an untried reachable setting within mean ± 2*std and public limits.

    The caller has already tested its point prediction. Targets are ordered by
    the minimum number of offered adjustments, then distance to the prediction
    mean; exact ties choose the smaller setting. Intermediate settings may have
    been visited while travelling to a new target. They still cost actions.
    Public range endpoints clip offered adjustments, as in native scalar control.
    """
    if max_actions <= 0 or not all(math.isfinite(value) for value in (current, mean, std)) or std < 0:
        return None
    control = public_scalar_control(public_dialog)
    if control is None:
        return None
    failed = {_decimal(value) for value in failed_values if math.isfinite(value)}
    if not failed:
        return None
    value, center, spread = map(_decimal, (current, mean, std))
    if not control.lower_hz <= value <= control.upper_hz:
        return None
    lower = max(control.lower_hz, center - 2 * spread)
    upper = min(control.upper_hz, center + 2 * spread)
    if lower > upper:
        return None
    # Breadth-first layers measure actual action count, including compositions
    # of coarse/fine public steps. Decimal arithmetic avoids duplicate decimal
    # settings produced through different adjustment paths.
    visited, frontier = {value}, {value}
    for _ in range(min(int(max_actions), MAX_PROBE_ACTIONS)):
        reached = set()
        for origin in frontier:
            for step in control.increments_hz:
                candidate = min(max(origin + step, control.lower_hz), control.upper_hz)
                if candidate not in visited:
                    reached.add(candidate)
        if not reached:
            return None
        eligible = [candidate for candidate in reached if lower <= candidate <= upper and candidate not in failed]
        if eligible:
            return float(min(eligible, key=lambda candidate: (abs(candidate - center), candidate)))
        visited.update(reached)
        frontier = reached
    return None


class ScalarProbe:
    """Per-target probe budget driven only by actual returned feedback.

    Initially inactive. Report the completed original point attempt with
    ``point_attempt=True``; its actions do not count as extra probes. Subsequent
    ``adjustment_actions`` count *all* actual tuning actions, including travel
    through previously failed settings. Observed activation ends probing.
    """
    def __init__(self, *, max_adjustment_actions: int = MAX_PROBE_ACTIONS):
        if not isinstance(max_adjustment_actions, int) or not 0 <= max_adjustment_actions <= MAX_PROBE_ACTIONS:
            raise ValueError("Extra scalar adjustments must be an integer from 0 to 16")
        self.max_adjustment_actions = max_adjustment_actions
        self.adjustment_actions = 0
        self.started = False
        self.activated = False
        self._failed_values = set()
        self._unconfirmed = False

    @property
    def failed_values(self) -> frozenset[float]:
        return frozenset(self._failed_values)

    @property
    def remaining_actions(self) -> int:
        return max(0, self.max_adjustment_actions - self.adjustment_actions)

    def observe_attempt(self, value: float, *, activated: bool | None,
                        point_attempt: bool = False, adjustment_actions: int = 1):
        """Record returned Hz/activation, not a requested but unexecuted target.

        Unknown activation does not establish a failed point. No next probe is
        suggested until an actual observation resolves that feedback.
        """
        if not math.isfinite(value) or not isinstance(adjustment_actions, int) or adjustment_actions < 0:
            raise ValueError("Expected finite returned Hz and a nonnegative actual adjustment count")
        if self.started:
            self.adjustment_actions += adjustment_actions
        self._unconfirmed = activated is None
        if activated is True:
            self.activated = True
        elif activated is False:
            self._failed_values.add(float(value))
            if point_attempt:
                self.started = True

    def next_target(self, current: float, mean: float, std: float, public_dialog: dict) -> float | None:
        if not self.started or self.activated or self._unconfirmed:
            return None
        return next_target(current, mean, std, public_dialog, self.failed_values,
                           max_actions=self.remaining_actions)
