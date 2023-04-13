"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
        12/21 - GND
"""
from typing import Optional

import mav_sim.parameters.control_parameters as AP
import numpy as np
from mav_sim.chap6.pd_control_with_rate import PDControlWithRate
from mav_sim.chap6.pid_control import PIDControl
from mav_sim.chap6.pi_control import PIControl
from mav_sim.chap6.my_pi_control import myPIControl
from mav_sim.chap6.tf_control import TFControl
from mav_sim.message_types.msg_autopilot import MsgAutopilot
from mav_sim.message_types.msg_delta import MsgDelta
from mav_sim.message_types.msg_state import MsgState
import mav_sim.parameters.aerosonde_parameters as MAV
# from mav_sim.tools.transfer_function import TransferFunction
from mav_sim.tools.wrap import saturate, wrap


class Autopilot:
    """Creates an autopilot for controlling the mav to desired values
    """
    def __init__(self, ts_control: float) -> None:
        """Initialize the lateral and longitudinal controllers

        Args:
            ts_control: time step for the control
        """

        # instantiate trajectory control 
        #part of equations 14.31 to 14.33
        self.un_PID=myPIControl(kp=AP.traj_kp_north,ki=AP.traj_ki_north,kd=AP.traj_kd_north,Ts=ts_control)
        self.ue_PID=myPIControl(kp=AP.traj_kp_east,ki=AP.traj_ki_east,kd=AP.traj_kd_east,Ts=ts_control)
        self.ud_PID=myPIControl(kp=AP.traj_kp_down,ki=AP.traj_ki_down,kd=AP.traj_kd_down,Ts=ts_control)
        

        # instantiate Attitude control
        #parts of equations 14.34 to 14.36
        self.tx_PD=PDControlWithRate(kp=AP.att_kp_phi,kd=AP.att_kd_phi)
        self.ty_PD=PDControlWithRate(kp=AP.att_kp_theta,kd=AP.att_kd_theta)
        #use PI control and pass the deriv term
        self.tz_PID=myPIControl(kp=AP.att_kp_psi,ki=AP.att_ki_psi,kd=AP.att_kd_psi,Ts=ts_control)
        self.commanded_state = MsgState()

    def update(self, cmd: MsgAutopilot, state: MsgState) -> tuple[MsgDelta, MsgState]:
        """Given a state and autopilot command, compute the control to the mav

        Args:
            cmd: command to the autopilot
            state: current state of the mav

        Returns:
            delta: low-level flap commands
            commanded_state: the state being commanded
        """
        g=MAV.gravity
        m=MAV.mass

        #Equations on page 299

        # att autopilot

        #PHI_C #no wrap treat like height commnad not use wrap (use the saturation of height)
        u_east=self.ue_PID.update(state.east,cmd.east_command,state.u)
        u_north=self.un_PID.update(state.north,cmd.north_command,state.v)
        phi=((u_east/g)*np.cos(state.psi)-(u_north/g)*np.sin(state.psi))
        up=np.pi/6
        phi_c=saturate(phi,-up,up)

        #THETA_C
        u_east=self.ue_PID.update(state.east,cmd.east_command,state.u)
        u_north=self.un_PID.update(state.north,cmd.north_command,state.v)
        theta=(-(u_north/g)*np.cos(state.psi)-(u_east/g)*np.sin(state.psi))
        up=np.pi/6
        theta_c=saturate(theta,-up,up)

        #T_C
        u_down=self.ud_PID.update(-state.altitude,cmd.down_command,state.w)
        T=(m*(g-u_down))
        up=1
        T_C=saturate(T,-up,up)
        
        #PSI_C 
        psi_c=cmd.psi_command

        # traj autopilot

        #tx
        tx=self.tx_PD.update(phi_c,state.phi,state.p) #rewrap 
        ty=self.ty_PD.update(theta_c,state.theta,state.q)
        tz=self.tz_PID.update(psi_c,state.psi,state.r)

        # construct control outputs and commanded states
        delta = MsgDelta(T=T_C,
                         tx=tx,
                         ty=ty,
                         tz=tz)
        self.commanded_state.T = T_C
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.psi = psi_c
        self.commanded_state.north=cmd.north_command
        self.commanded_state.east=cmd.east_command
        self.commanded_state.altitude=-cmd.down_command

        return delta, self.commanded_state.copy()

        # phi_c = 0. # commanded value for phi
        # theta_c = 0. # commanded value for theta
        # delta_a = 0.
        # delta_r = 0.

        # delta_e = 0.
        # delta_t = 0.
        # delta_t = saturate(delta_t, 0.0, 1.0)