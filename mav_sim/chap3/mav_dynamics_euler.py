"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV using Euler coordinates
    - use Euler angles for the attitude state

part of mavsimPy
    - Beard & McLain, PUP, 2012
    - Update history:
        12/17/2018 - RWB
        1/14/2019 - RWB
        12/21 - GND
        12/22 - GND
"""
import mav_sim.parameters.aerosonde_parameters as MAV
import numpy as np
from mav_sim.chap3.mav_dynamics import IND  , ForceMoments
from mav_sim.tools import types
from mav_sim.tools.rotations import Euler2Quaternion, Euler2Rotation, Quaternion2Euler


# Indexing constants for state using Euler representation
class StateIndicesEuler:
    """Constant class for easy access of state indices
    """
    NORTH: int  = 0  # North position
    EAST: int   = 1  # East position
    DOWN: int   = 2  # Down position
    U: int      = 3  # body-x velocity
    V: int      = 4  # body-y velocity
    W: int      = 5  # body-z velocity
    PHI: int    = 6  # Roll angle (about x-axis)
    THETA: int  = 7  # Pitch angle (about y-axis)
    PSI: int    = 8  # Yaw angle (about z-axis)
    P: int      = 9 # roll rate - body frame - i
    Q: int      = 10 # pitch rate - body frame - j
    R: int      = 11 # yaw rate - body frame - k
    VEL: list[int] = [U, V, W] # Body velocity indices
    ANG_VEL: list[int] = [P, Q, R] # Body rotational velocities
    NUM_STATES: int = 12 # Number of states
IND_EULER = StateIndicesEuler()

# Conversion functions
def euler_state_to_quat_state(state_euler: types.DynamicStateEuler) -> types.DynamicState:
    """Converts an Euler state representation to a quaternion state representation

    Args:
        state_euler: The state vector to be converted to a quaternion representation

    Returns:
        state_quat: The converted state
    """
    # Create the quaternion from the euler coordinates
    e = Euler2Quaternion(phi=state_euler[IND_EULER.PHI], theta=state_euler[IND_EULER.THETA], psi=state_euler[IND_EULER.PSI])

    # Copy over data
    state_quat = np.zeros((IND.NUM_STATES,1))
    state_quat[IND.NORTH] = state_euler.item(IND_EULER.NORTH)
    state_quat[IND.EAST] = state_euler.item(IND_EULER.EAST)
    state_quat[IND.DOWN] = state_euler.item(IND_EULER.DOWN)
    state_quat[IND.U] = state_euler.item(IND_EULER.U)
    state_quat[IND.V] = state_euler.item(IND_EULER.V)
    state_quat[IND.W] = state_euler.item(IND_EULER.W)
    state_quat[IND.E0] = e.item(0)
    state_quat[IND.E1] = e.item(1)
    state_quat[IND.E2] = e.item(2)
    state_quat[IND.E3] = e.item(3)
    state_quat[IND.P] = state_euler.item(IND_EULER.P)
    state_quat[IND.Q] = state_euler.item(IND_EULER.Q)
    state_quat[IND.R] = state_euler.item(IND_EULER.R)

    return state_quat

def quat_state_to_euler_state(state_quat: types.DynamicState) -> types.DynamicStateEuler:
    """Converts a quaternion state representation to an Euler state representation

    Args:
        state_quat: The state vector to be converted

    Returns
        state_euler: The converted state
    """
    # Create the quaternion from the euler coordinates
    phi, theta, psi = Quaternion2Euler(state_quat[IND.QUAT])

    # Copy over data
    state_euler = np.zeros((IND_EULER.NUM_STATES,1))
    state_euler[IND_EULER.NORTH] = state_quat.item(IND.NORTH)
    state_euler[IND_EULER.EAST]  = state_quat.item(IND.EAST)
    state_euler[IND_EULER.DOWN]  = state_quat.item(IND.DOWN)
    state_euler[IND_EULER.U]     = state_quat.item(IND.U)
    state_euler[IND_EULER.V]     = state_quat.item(IND.V)
    state_euler[IND_EULER.W]     = state_quat.item(IND.W)
    state_euler[IND_EULER.PHI]   = phi
    state_euler[IND_EULER.THETA] = theta
    state_euler[IND_EULER.PSI]   = psi
    state_euler[IND_EULER.P]     = state_quat.item(IND.P)
    state_euler[IND_EULER.Q]     = state_quat.item(IND.Q)
    state_euler[IND_EULER.R]     = state_quat.item(IND.R)

    return state_euler


class DynamicStateEuler:
    """Struct for the dynamic state
    """
    def __init__(self, state: types.DynamicStateEuler ) -> None:
        self.north: float     # North position
        self.east: float      # East position
        self.down: float      # Down position
        self.u: float         # body-x velocity
        self.v: float         # body-y velocity
        self.w: float         # body-z velocity
        self.phi: float       # roll angle (about x-axis)
        self.theta: float     # pitch angle (about y-axis)
        self.psi: float       # yaw angle (about z-axis)
        self.p: float         # roll rate - body frame - i
        self.q: float         # pitch rate - body frame - j
        self.r: float         # yaw rate - body frame - k

        self.extract_state(state)

    def extract_state(self, state: types.DynamicStateEuler) ->None:
        """Initializes the state variables

        Args:
            state: State from which to extract the state values

        """
        self.north = state.item(IND_EULER.NORTH)
        self.east =  state.item(IND_EULER.EAST)
        self.down =  state.item(IND_EULER.DOWN)
        self.u =     state.item(IND_EULER.U)
        self.v =     state.item(IND_EULER.V)
        self.w =     state.item(IND_EULER.W)
        self.phi =   state.item(IND_EULER.PHI)
        self.theta = state.item(IND_EULER.THETA)
        self.psi =   state.item(IND_EULER.PSI)
        self.p =     state.item(IND_EULER.P)
        self.q =     state.item(IND_EULER.Q)
        self.r =     state.item(IND_EULER.R)

    def convert_to_numpy(self) -> types.DynamicStateEuler:
        """Converts the state to a numpy object
        """
        output = np.empty( (IND_EULER.NUM_STATES,1) )
        output[IND_EULER.NORTH, 0] = self.north
        output[IND_EULER.EAST, 0] = self.east
        output[IND_EULER.DOWN, 0] = self.down
        output[IND_EULER.U, 0] = self.u
        output[IND_EULER.V, 0] = self.v
        output[IND_EULER.W, 0] = self.w
        output[IND_EULER.PHI, 0] = self.phi
        output[IND_EULER.THETA, 0] = self.theta
        output[IND_EULER.PSI, 0] = self.psi
        output[IND_EULER.P, 0] = self.p
        output[IND_EULER.Q, 0] = self.q
        output[IND_EULER.R, 0] = self.r

        return types.DynamicState( output )

def derivatives_euler(state: types.DynamicStateEuler, forces_moments: types.ForceMoment) -> types.DynamicStateEuler:
    """Implements the dynamics xdot = f(x, u) where u is the force/moment vector

    Args:
        state: Current state of the vehicle
        forces_moments: 6x1 array containing [fx, fy, fz, Mx, My, Mz]^T

    Returns:
        Time derivative of the state ( f(x,u), where u is the force/moment vector )
    """
    st=DynamicStateEuler(state)
    fm=ForceMoments(forces_moments)

    #Declare some constants 
    mass=MAV.mass
    jx=MAV.Jx
    jy=MAV.Jy
    jz=MAV.Jz
    angles=state[IND_EULER.ANG_VEL]
    l=fm.l
    m=fm.m
    n=fm.n

    #initialize variables
    x_dot = np.empty( (IND_EULER.NUM_STATES,1) )
    pos=np.array([[st.u],[st.v],[st.w]])  
    
    rot_mat=Euler2Rotation(st.phi,st.theta,st.psi)
    p_dot=rot_mat@pos


    #Implement equations from pg 286 with slight mod
    x_dot[IND_EULER.NORTH] =p_dot.item(0)
    x_dot[IND_EULER.EAST] =p_dot.item(1)
    x_dot[IND_EULER.DOWN] =p_dot.item(2)

    x_dot[IND_EULER.U]= (1/mass)*(fm.fx)                                            
    x_dot[IND_EULER.V]= (1/mass)*(fm.fy)
    x_dot[IND_EULER.W]= (1/mass)*(fm.fz)

    angle_mat=np.array([[1., np.sin(state.item(IND_EULER.PHI))*np.tan(state.item(IND_EULER.THETA)), \
                         np.cos(state.item(IND_EULER.PHI))*np.tan(state.item(IND_EULER.THETA))],
                        [0., np.cos(state.item(IND_EULER.PHI)), -np.sin(state.item(IND_EULER.PHI))],
                        [0., (np.sin(state.item(IND_EULER.PHI))/np.cos(state.item(IND_EULER.THETA))), \
                            (np.cos(state.item(IND_EULER.PHI))/np.cos(state.item(IND_EULER.THETA)))]
                        ])

    a_m=angle_mat@state[IND_EULER.ANG_VEL]

    x_dot[IND_EULER.PHI]   = a_m[0]
    x_dot[IND_EULER.THETA] = a_m[1]
    x_dot[IND_EULER.PSI]   = a_m[2]
    

    x_dot[IND_EULER.P]=((jy-jz)/jx)*(st.q*st.r)+(l/jx)
    x_dot[IND_EULER.Q]=((jz-jx)/jy)*(st.p*st.r)+(m/jy)
    x_dot[IND_EULER.R]=((jx-jy)/jz)*(st.p*st.q)+(n/jz)

    #>>>>>>>>>>>>>>>>>>>>>>>>>>>PREVIOUS WORK<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # rot_mat=Euler2Rotation(state.item(IND_EULER.PHI),state.item(IND_EULER.THETA),state.item(IND_EULER.PSI))
    # r_e=rot_mat@np.array([state[IND_EULER.U],state[IND_EULER.V],state[IND_EULER.W]])
    
    # # # # # collect the derivative of the states
    # x_dot = np.empty( (IND_EULER.NUM_STATES,1) )

    # x_dot[IND_EULER.NORTH] =r_e[0]
    # x_dot[IND_EULER.EAST] = r_e[1]
    # x_dot[IND_EULER.DOWN] = r_e[2]
    # #print(state)
    # x_dot[IND_EULER.U] = (state[IND_EULER.R]*state[IND_EULER.V]-state[IND_EULER.Q]*state[IND_EULER.W])\
    #     +(1/MAV.mass)*(forces_moments[0])
    # x_dot[IND_EULER.V] = (state[IND_EULER.P]*state[IND_EULER.W]-state[IND_EULER.R]*state[IND_EULER.U])\
    #     +(1/MAV.mass)*(forces_moments[1])
    # x_dot[IND_EULER.W] = (state[IND_EULER.Q]*state[IND_EULER.U]-state[IND_EULER.P]*state[IND_EULER.V])\
    #     +(1/MAV.mass)*(forces_moments[2])

    # angle_mat=np.array([[1., np.sin(state.item(IND_EULER.PHI))*np.tan(state.item(IND_EULER.THETA)), \
    #                      np.cos(state.item(IND_EULER.PHI))*np.tan(state.item(IND_EULER.THETA))],
    #                     [0., np.cos(state.item(IND_EULER.PHI)), -np.sin(state.item(IND_EULER.PHI))],
    #                     [0., (np.sin(state.item(IND_EULER.PHI))/np.cos(state.item(IND_EULER.THETA))), \
    #                         (np.cos(state.item(IND_EULER.PHI))/np.cos(state.item(IND_EULER.THETA)))]
    #                     ])

    # a_m=angle_mat@state[IND_EULER.ANG_VEL]

    # x_dot[IND_EULER.PHI]   = a_m[0]
    # x_dot[IND_EULER.THETA] = a_m[1]
    # x_dot[IND_EULER.PSI]   = a_m[2]

    # x_dot[IND_EULER.P] = (MAV.gamma1*state[IND_EULER.P]*state[IND_EULER.Q]-MAV.gamma2*state[IND_EULER.Q]*state[IND_EULER.R])\
    #     +(MAV.gamma3*forces_moments[3]+MAV.gamma4*forces_moments[5])
    # x_dot[IND_EULER.Q] = (MAV.gamma5*state[IND_EULER.P]*state[IND_EULER.R]\
    #     -MAV.gamma6*(state[IND_EULER.P]**2-state[IND_EULER.R]**2))\
    #     +((1/(MAV.Jy))*forces_moments[4])
    # x_dot[IND_EULER.R] = (MAV.gamma7*state[IND_EULER.P]*state[IND_EULER.Q]-MAV.gamma1*state[IND_EULER.Q]*state[IND_EULER.R])\
    #     +(MAV.gamma4*forces_moments[3]+MAV.gamma8*forces_moments[5])
    return x_dot
