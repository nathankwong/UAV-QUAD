"""
msg_delta
    - messages type for input to the aircraft

part of mavsim_python
    - Beard & McLain, PUP, 2012
    - Last update:
        2/27/2020 - RWB
        12/2021 - GND
"""
from typing import Any, Type, TypeVar

import numpy as np
import numpy.typing as npt

Entity = TypeVar('Entity', bound='MsgDelta')
class MsgDelta:
    """Message inputs for the aircraft
    """
    def __init__(self,
                 T: float =0.00,
                 tx: float =0.00,
                 ty: float =0.00,
                 tz: float =0.0
                 ) -> None:
        """Set the commands to default values
        """
        self.T: float = T  # omega 1 command
        self.tx: float = tx  # omega 2 command
        self.ty: float = ty  # omega 3 command
        self.tz: float = tz  # omega 4 command

    def copy(self, msg: Type[Entity]) -> None:
        """
        Initializes the command message from the input
        """
        self.T = msg.T
        self.tx = msg.tx
        self.ty = msg.ty
        self.tz = msg.tz

    def to_array(self) -> npt.NDArray[Any]:
        """Convert the command to a numpy array
        """
        return np.array([[self.T],
                         [self.tx],
                         [self.ty],
                         [self.tz]])

    def from_array(self, u: npt.NDArray[Any]) -> None:
        """Extract the commands from a numpy array
        """
        self.T = u.item(0)
        self.tx = u.item(1)
        self.ty = u.item(2)
        self.tz = u.item(3)

    def print(self) -> None:
        """Print the commands to the console
        """
        print('elevator=', self.T,
              'aileron=', self.tx,
              'rudder=', self.ty,
              'throttle=', self.tz)

    def __str__(self) -> str:
        """Create a string from the commands"""
        out = 'elevator=' + str(self.T) + \
              ', aileron=' + str(self.tx) + \
              ', rudder=' + str(self.ty) + \
              ', throttle=' + str(self.tz)
        return out
