"""
mavsim_python
    - Chapter 6 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/5/2019 - RWB
        2/24/2020 - RWB
"""


import numpy as np
from mav_sim.chap3.mav_dynamics import DynamicState
from mav_sim.chap6.run_sim import run_sim
from mav_sim.message_types.msg_sim_params import MsgSimParams
from mav_sim.tools.signals import Signals


def main() -> None:
    """Provide a test scenario
    """
    # Initialize the simulation parameters
    sim_params = MsgSimParams(end_time=100., video_name="chap6.avi") # Sim ending in 10 seconds
    state = DynamicState()

    # Calculate control signals
    psi_cmd = Signals(dc_offset=np.radians(90),
                        amplitude=1,
                        start_time=0,
                        frequency=0.01)
    
    north_cmd = Signals(dc_offset=100.0,
                            amplitude=20.0,
                            start_time=0.0,
                            frequency=0.02)
    
    east_cmd = Signals(dc_offset=100.0,
                            amplitude=20.0,
                            start_time=0.0,
                            frequency=0.02)
    
    down_cmd = Signals(dc_offset=-100.0,
                            amplitude=20.0,
                            start_time=0.0,
                            frequency=0.025)

    # Run the simulation - Note that while not used, the viewer objects
    # need to remain active to keep the windows open
    (mav_view, data_view) = run_sim(sim=sim_params, init_state=state, \
        psi_cmd=psi_cmd, north_cmd=north_cmd, east_cmd=east_cmd,down_cmd=down_cmd) #pylint: disable=unused-variable

    # Wait until user is finished
    print("Press any key to close")
    input()

if __name__ == "__main__":
    main()
