"""
mavDynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

part of mavPySim
    - Beard & McLain, PUP, 2012
    - Update history:
        12/20/2018 - RWB
"""
from typing import Optional, cast

import mav_sim.parameters.aerosonde_parameters as MAV
import numpy as np

# load mav dynamics from previous chapter
from mav_sim.chap3.mav_dynamics import IND, DynamicState, ForceMoments, derivatives
from mav_sim.message_types.msg_delta import MsgDelta

# load message types
from mav_sim.message_types.msg_state import MsgState
from mav_sim.tools import types
from mav_sim.tools.rotations import Quaternion2Euler, Quaternion2Rotation


class MavDynamics:
    """Implements the dynamics of the MAV using vehicle inputs and wind
    """

    def __init__(self, Ts: float, state: Optional[DynamicState] = None):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        if state is None:
            self._state = DynamicState().convert_to_numpy()
        else:
            self._state = state.convert_to_numpy()

        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec

        # update velocity data
        (self._Va, self._alpha, self._beta, self._wind) = update_velocity_data(self._state)

        # Update forces and moments data
        self._forces = np.array([[0.], [0.], [0.]]) # store forces to avoid recalculation in the sensors function (ch 7)
        self._moments = np.array([[0.], [0.], [0.]]) # store moments to avoid recalculation
        forces_moments_vec = forces_moments(self._state, MsgDelta(), self._Va)
        self._forces[0] = forces_moments_vec.item(0)
        self._forces[1] = forces_moments_vec.item(1)
        self._forces[2] = forces_moments_vec.item(2)
        self._moments[0] = forces_moments_vec.item(3)
        self._moments[1] = forces_moments_vec.item(4)
        self._moments[2] = forces_moments_vec.item(5)


        # initialize true_state message
        self.true_state = MsgState()
        self._update_true_state()

    ###################################
    # public functions
    def update(self, delta: MsgDelta, wind: types.WindVector, time_step: Optional[float] = None) -> None:
        """
        Integrate the differential equations defining dynamics, update sensors

        Args:
            delta : (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind: the wind vector in inertial coordinates
        """
        # get forces and moments acting on rigid bod
        forces_moments_vec = forces_moments(self._state, delta, self._Va)
        self._forces[0] = forces_moments_vec.item(0)
        self._forces[1] = forces_moments_vec.item(1)
        self._forces[2] = forces_moments_vec.item(2)
        self._moments[0] = forces_moments_vec.item(3)
        self._moments[1] = forces_moments_vec.item(4)
        self._moments[2] = forces_moments_vec.item(5)

        # Get the timestep
        if time_step is None:
            time_step = self._ts_simulation

        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = derivatives(self._state, forces_moments_vec)
        k2 = derivatives(self._state + time_step/2.*k1, forces_moments_vec)
        k3 = derivatives(self._state + time_step/2.*k2, forces_moments_vec)
        k4 = derivatives(self._state + time_step*k3, forces_moments_vec)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(IND.E0)
        e1 = self._state.item(IND.E1)
        e2 = self._state.item(IND.E2)
        e3 = self._state.item(IND.E3)
        norm_e = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[IND.E0][0] = self._state.item(IND.E0)/norm_e
        self._state[IND.E1][0] = self._state.item(IND.E1)/norm_e
        self._state[IND.E2][0] = self._state.item(IND.E2)/norm_e
        self._state[IND.E3][0] = self._state.item(IND.E3)/norm_e

        # update the airspeed, angle of attack, and side slip angles using new state
        (self._Va, self._alpha, self._beta, self._wind) = update_velocity_data(self._state, wind)

        # update the message class for the true state
        self._update_true_state()

    def external_set_state(self, new_state: types.DynamicState) -> None:
        """Loads a new state
        """
        self._state = new_state

    def get_state(self) -> types.DynamicState:
        """Returns the state
        """
        return self._state

    def get_struct_state(self) ->DynamicState:
        '''Returns the current state in a struct format

        Outputs:
            DynamicState: The latest state of the mav
        '''
        return DynamicState(self._state)

    def get_fm_struct(self) -> ForceMoments:
        '''Returns the latest forces and moments calculated in dynamic update'''
        force_moment = np.zeros((6,1))
        force_moment[0:3] = self._forces
        force_moment[3:6] = self._moments
        return ForceMoments(force_moment= cast(types.ForceMoment, force_moment) )

    def get_euler(self) -> tuple[float, float, float]:
        '''Returns the roll, pitch, and yaw Euler angles based upon the state'''
        # Get Euler angles
        phi, theta, psi = Quaternion2Euler(self._state[IND.QUAT])

        # Return angles
        return (phi, theta, psi)

    ###################################
    # private functions
    def _update_true_state(self) -> None:
        """ update the class structure for the true state:

        [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        """
        quat = cast(types.Quaternion, self._state[IND.QUAT])
        phi, theta, psi = Quaternion2Euler(quat)
        pdot = Quaternion2Rotation(quat) @ self._state[IND.VEL]
        self.true_state.north = self._state.item(IND.NORTH)
        self.true_state.east = self._state.item(IND.EAST)
        self.true_state.altitude = -self._state.item(IND.DOWN)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = cast(float, np.linalg.norm(pdot))
        if self.true_state.Vg != 0.:
            self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        else:
            self.true_state.gamma = 0.
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(IND.P)
        self.true_state.q = self._state.item(IND.Q)
        self.true_state.r = self._state.item(IND.R)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)

def skew(vector: types.NP_MAT) -> types.NP_MAT: 
    """
    Returns a skewed matrix. Referance p.287

    Args: 
        vector: A quaterion vector
    Returns: 
        A skewed matrix
    """
    a=vector.item(1)
    b=vector.item(2)
    c=vector.item(3)
    
    mat=np.array([[0, -c, b],
                  [c , 0, -a],
                  [-b, a, 0]])
    
    return cast(types.NP_MAT,mat)

def rotation(q:types.NP_MAT)-> types.NP_MAT: 
    """
    Rotation Matrix using the skew function. Referance p.287

    Args:
        q: A quaterion vector
    Returns:
        R: Matrix that does the following: R=I+2*q0*skew(q)+2*skew(q)@skew(q)
    """
    q0=q.item(0)
    I=np.identity(3)

    R=I+2*q0*skew(q)+2*skew(q)@skew(q)

    return cast(types.NP_MAT,R)

def forces_moments(state: types.DynamicState, delta: MsgDelta, Va: float) -> types.ForceMoment:
    """
    Return the forces on the UAV based on the state, wind, and control surfaces

    Args:
        state: current state of the aircraft
        delta: flap and thrust commands
        Va: Airspeed
        beta: Side slip angle
        alpha: Angle of attack


    Returns:
        Forces and Moments on the UAV (in body frame) np.matrix(fx, fy, fz, Mx, My, Mz)
    """
    # Extract elements
    st=DynamicState()
    #Prop Force

    # w1, w2,w3,w4=motor_thrust_torque(delta=delta)

    #coeff

    mass=MAV.mass
    g=MAV.gravity
    D=MAV.D_prop
    C_d=MAV.C_D
    q=state[IND.QUAT]
    e3=np.array([[0],[0],[1]])
    
    x=np.array([1,1,0])
    D=g*C_d*np.diag(x)

    v=np.array([[st.u],[st.u],[st.w]])

    #forces
    f_g=mass*g
    f_ind=-mass*rotation(q=q)@D@np.transpose(rotation(q=q))@v
    f_T=-delta.T*rotation(q=q)@e3

    #Quadcopter Forces
    fx = f_ind.item(0)+f_T.item(0)
    fy = f_ind.item(1)+f_T.item(1)
    fz = f_ind.item(2)+f_T.item(2)+f_g
        
    Mx = delta.tx
    My = delta.ty
    Mz = delta.tz

    # print('mav_dynamics::forces_moments() Needs to be implemented')
    return types.ForceMoment( np.array([[fx, fy, fz, Mx, My, Mz]]).T )

def motor_thrust_torque(delta: MsgDelta) -> tuple[float, float, float,float ]:
    """ compute thrust and torque due to propeller  (See addendum by McLain)

    Args:
        Va: Airspeed
        delta_t: Throttle command

    Returns:
        T_p: Propeller thrust
        Q_p: Propeller torque
    """
    #Coeffs and constants
    C_T=0.55
    C_Q=0.24

    D=MAV.D_prop
    L=MAV.Length
    root=np.sqrt(2)/2

    # Trust and Torque for T matrix
    thrust_prop   = delta.T
    torque_prop_x = delta.tx
    torque_prop_y = delta.ty
    torque_prop_z = delta.tz

    #mixing matrix from page 292

    M=np.array([[C_T, C_T, C_T,C_T],
                [-C_Q*D*L*(root),-C_Q*D*L*(root),C_Q*D*L*(root),C_Q*D*L*(root)],
                [C_Q*D*L*(root),-C_Q*D*L*(root),-C_Q*D*L*(root),C_Q*D*L*(root)],
                [C_Q*D,-C_Q*D,C_Q*D,-C_Q*D]])
    
    #omega vector from page 292
    omega=np.linalg.inv(M)@np.array([[thrust_prop],[torque_prop_x],[torque_prop_y],[torque_prop_z]])

    Ohm_OP1= omega.item(0)
    Ohm_OP2= omega.item(1)
    Ohm_OP3= omega.item(2)
    Ohm_OP4= omega.item(3)

    return np.sqrt(Ohm_OP1), np.sqrt(Ohm_OP2),np.sqrt(Ohm_OP3),np.sqrt(Ohm_OP4)

#Scrach 
    # print('mav_dynamics::motor_thrust_torque() Needs to be implemented')
    # a = MAV.rho*MAV.D_prop**5/(4*np.pi**2)*MAV.C_Q0     
    # b = MAV.rho*MAV.D_prop**4/(2*np.pi)*MAV.C_Q1*Va + MAV.KQ/MAV.R_motor(30/(np.pi*MAV.KV))     

    # c1 = MAV.rho*MAV.D_prop**3*MAV.C_Q2*Va**2 - MAV.KQ*MAV.V_max*delta_t/MAV.R_motor + MAV.KQ*MAV.i0  
    # c2 = MAV.rho*MAV.D_prop**3*MAV.C_Q2*Va**2 - MAV.KQ*MAV.V_max*delta_t/MAV.R_motor + MAV.KQ*MAV.i0     
    # c3 = MAV.rho*MAV.D_prop**3*MAV.C_Q2*Va**2 - MAV.KQ*MAV.V_max*delta_t/MAV.R_motor + MAV.KQ*MAV.i0     
    # c4 = MAV.rho*MAV.D_prop**3*MAV.C_Q2*Va**2 - MAV.KQ*MAV.V_max*delta_t/MAV.R_motor + MAV.KQ*MAV.i0 

    # Ohm_OP1 = (-b + np.sqrt(b**2 - 4*a*c1))/(2*a)     
    # Ohm_OP2 = (-b + np.sqrt(b**2 - 4*a*c2))/(2*a)  
    # Ohm_OP3 = (-b + np.sqrt(b**2 - 4*a*c3))/(2*a)  
    # Ohm_OP4 = (-b + np.sqrt(b**2 - 4*a*c4))/(2*a)  

    # if Ohm_OP1 == 0:
    #     J1=0
    # else: 
    #     J1 = 2*np.pi*Va/(Ohm_OP1*MAV.D_prop)    

    # if Ohm_OP2 == 0:
    #     J2=0
    # else: 
    #     J2 = 2*np.pi*Va/(Ohm_OP2*MAV.D_prop)   

    # if Ohm_OP3 == 0:
    #     J3=0
    # else: 
    #     J3 = 2*np.pi*Va/(Ohm_OP3*MAV.D_prop) 

    # if Ohm_OP4 == 0:
    #     J4=0
    # else: 
    #     J4 = 2*np.pi*Va/(Ohm_OP4*MAV.D_prop)   


    # C_T1 = MAV.C_T2*J1**2 + MAV.C_T1*J1 + MAV.C_T0
    # C_T2 = MAV.C_T2*J2**2 + MAV.C_T1*J2 + MAV.C_T0
    # C_T3 = MAV.C_T2*J3**2 + MAV.C_T1*J3 + MAV.C_T0
    # C_T4 = MAV.C_T2*J4**2 + MAV.C_T1*J4 + MAV.C_T0     

    # C_Q1 = MAV.C_Q2*J1**2 + MAV.C_Q1*J2 + MAV.C_Q0
    # C_Q2 = MAV.C_Q2*J2**2 + MAV.C_Q1*J2 + MAV.C_Q0
    # C_Q3 = MAV.C_Q2*J3**2 + MAV.C_Q1*J3 + MAV.C_Q0
    # C_Q4 = MAV.C_Q2*J4**2 + MAV.C_Q1*J4 + MAV.C_Q0



    # M=np.array([[C_T1, C_T2, C_T3,C_T4],
    #             [-C_Q1*D*L*(root),-C_Q2*D*L*(root),C_Q3*D*L*(root),C_Q4*D*L*(root)],
    #             [C_Q1*D*L*(root),-C_Q2*D*L*(root),-C_Q3*D*L*(root),C_Q4*D*L*(root)],
    #             [C_Q1*D,-C_Q2*D,C_Q3*D,-C_Q4*D]])
    

def update_velocity_data(state: types.DynamicState,\
    wind: types.WindVector = types.WindVector( np.zeros((6,1)) ) \
    )  -> tuple[float, float, float, types.NP_MAT]:
    """Calculates airspeed, angle of attack, sideslip, and velocity wrt wind

    Args:
        state: current state of the aircraft

    Returns:
        Va: Airspeed
        alpha: Angle of attack
        beta: Side slip angle
        wind_inertial_frame: Wind vector in inertial frame
    """
    steady_state = wind[0:3]
    gust = wind[3:6]

    # convert wind vector from world to body frame
    R = Quaternion2Rotation(state[IND.QUAT]) # rotation from body to world frame
    wind_body_frame = R.T @ steady_state  # rotate steady state wind to body frame
    wind_body_frame += gust  # add the gust
    wind_inertial_frame = R @ wind_body_frame # Wind in the world frame
    st=DynamicState(state)
    V = np.array([[st.u - wind_body_frame[0]], [st.v - wind_body_frame[1]], [st.w - wind_body_frame[2]]])
    ur=V.item(0)
    vr=V.item(1)
    wr=V.item(2)
    
    # compute airspeed
    Va = float(np.sqrt(ur**2+vr**2+wr**2))

    # compute angle of attack
    alpha = float(np.arctan2(wr,ur))

    # compute sideslip angle
    if Va==0: 
        beta = float(np.arctan2(vr,ur**2+wr**2))
    else: 
        beta = float(np.arcsin(vr/Va))

   # Return computed values
   # print('mav_dynamics::update_velocity_data() Needs to be implemented')
    return (Va, alpha, beta, wind_inertial_frame)
