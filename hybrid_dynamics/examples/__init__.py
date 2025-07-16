"""Example hybrid systems."""

from .bipedal import BipedalWalker
from .bouncing_ball import BouncingBall
from .chua_circuit import ChuaCircuit
from .rimless_wheel import RimlessWheel
from .thermostat import Thermostat
from .unstableperiodic import UnstablePeriodicSystem

__all__ = [
    "BouncingBall",
    "Thermostat",
    "RimlessWheel",
    "UnstablePeriodicSystem",
    "BipedalWalker",
    "ChuaCircuit",
]
