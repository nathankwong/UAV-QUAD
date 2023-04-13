"""
msg_autopilot
    - messages type for input to the autopilot

part of mavsim_python
    - Beard & McLain, PUP, 2012
    - Last update:
        2/5/2019 - RWB
"""
from typing import Any

import numpy as np
import numpy.typing as npt


class MsgAutopilot:
    """Message class for commanding the autopilot."""

    __slots__ = [
        "north_command",
        "east_command",
        "down_command",
        "psi_command",
        "phi_command",
        "theta_command",
        "thrust_command",
    ]   

    def __init__(self) -> None:
        """Default parameters to zero"""
        self.north_command: float = 0.0  # commanded airspeed m/s
        self.east_command: float = 0.0  # commanded course angle in rad
        self.down_command: float = 0.0  # commanded altitude in m
        self.psi_command: float = 0.0  # feedforward command for roll angle
        self.phi_command: float=0.0
        self.theta_command: float=0.0
        self.thrust_command: float=0.0

    def to_array(self) -> npt.NDArray[Any]:
        """Convert the command to a numpy array."""
        return np.array(
            [
                [self.north_command],
                [self.east_command],
                [self.down_command],
                [self.psi_command],
                [self.phi_command],
                [self.theta_command],
                [self.thrust_command],
            ],
            dtype=float,
        )

    def __str__(self) -> str:
        """Returns a string of the class"""
        out =   "north_command: " + str(self.north_command) + \
                ", east_command: " + str(self.east_command) + \
                ", down_command: " + str(self.down_command) +\
                ", psi_command: " + str(self.psi_command) +\
                ", phi_command: " + str(self.phi_command) +\
                ", theta_command: " + str(self.theta_command) +\
                ", thrust_command: " + str(self.thrust_command) 
        return out
