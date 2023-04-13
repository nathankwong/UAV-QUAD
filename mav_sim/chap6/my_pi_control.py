"""
pi_control
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
        1/2022 - GND
"""
import numpy as np
from mav_sim.tools.wrap import saturate


class myPIControl:
    """Proportional integral controller
    """
    def __init__(self, kp: float =0.0, ki: float =0.0, kd: float =0.0, Ts: float =0.01, limit: float =1.0) -> None:
        """Store control gains

        Args:
            kp: proportional gain
            kd: derivative gain
            ki: integral gain
            Ts: update period
            limit: maximum magnitude for the control
        """
        self.kp = kp
        self.ki = ki
        self.kd =  kd
        self.Ts = Ts
        self.limit = limit
        self.integrator: float = 0.0
        self.error_delay_1: float = 0.0

    def update(self, y_ref: float, y: float, ydot: float) -> float:
        """Compute the proportional integral control given the reference and actual value.
           The control is saturated based upon self.limit and a simple anti-windup procedure
           is performed on the integrator.

        Args:
            y_ref: desired value
            y: actual value
            ydot: derivative of actual value
        Returns:
            u_sat: the saturated control input
        """

        # compute the error
        error =y_ref-y

        # Do something with the error
        self.integrator=self.integrator\
            +(self.Ts/2)*(error+self.error_delay_1)
        u = self.kp*error\
            +self.ki*self.integrator-self.kd*ydot
        u_sat = saturate(u,-self.limit,self.limit)

        # integral anti-windup
        if np.abs(self.ki)>0.0001:
            self.integrator=self.integrator\
                +(self.Ts/self.ki)*(u_sat-u)

        # update the delayed variables
        self.error_delay_1 = error
        return u_sat
