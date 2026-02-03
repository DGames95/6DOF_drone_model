"""
Quadcopter Simulation with Independent Motor Modelling
Each motor is modelled individually with position-based moment generation.
Body rates emerge from motor forces and geometry.

State: x = [p, v, λ, Ω, ω]^T (16 states)
    - p: position in inertial frame (3)
    - v: velocity in inertial frame (3)
    - λ: Euler angles [roll, pitch, yaw] (3)
    - Ω: body rates [p, q, r] (3)
    - ω: individual motor speeds [ω1, ω2, ω3, ω4] (4)
"""

import numpy as np
from scipy.integrate import odeint
import control as ct
from pathlib import Path

from params import params, motor_params


class QuadcopterIndependentMotors:
    """
    Quadcopter with per-motor dynamics.
    
    Motors are placed at distinct positions relative to CoM.
    Each motor produces:
    - Thrust: along negative z-body
    - Torque: reaction torque opposing spin
    - Moment arms: position × force creates attitude moments
    """
    
    def __init__(self, params_dict, motor_params_dict):
        """Initialize with global and motor-specific parameters"""
        
        self.params = params_dict
        self.motor_params = motor_params_dict
        self.g = 9.81
        
        # Inertia
        self.Jx = params_dict['Jx']
        self.Jy = params_dict['Jy']
        self.Jz = params_dict['Jz']
        self.J = np.diag([self.Jx, self.Jy, self.Jz])
        
        # Motor configuration
        self.motor_positions = np.array(motor_params_dict['motor_positions'])
        self.k_thrust = np.array(motor_params_dict['k_thrust'])
        self.k_torque = np.array(motor_params_dict['k_torque'])
        self.motor_directions = np.array(motor_params_dict['motor_directions'])
        self.motor_inertia = np.array(motor_params_dict['motor_inertia'])
        
        # Drag coefficients
        self.k_drag_linear = motor_params_dict['k_drag_linear']
        self.k_drag_quad_xy = motor_params_dict['k_drag_quad_xy']
        self.k_drag_quad_z = motor_params_dict['k_drag_quad_z']
        
        # Motor dynamics
        self.k_motor = motor_params_dict['k_motor']
        self.tau_motor = motor_params_dict['tau_motor']
        self.w_min = params_dict['w_min']
        self.w_max = params_dict['w_max']
        
        # Number of motors
        self.n_motors = len(self.motor_positions)
    
    def rotation_matrix_zyx(self, euler):
        """Compute rotation matrix from Euler angles (ZYX convention)"""
        phi, theta, psi = euler
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx
    
    def body_rate_to_euler_rate(self, euler, body_rates):
        """Convert body rates to Euler angle rates"""
        phi, theta, psi = euler
        p, q, r = body_rates
        
        Q = np.array([
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
            [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
        ])
        
        euler_rates = Q @ np.array([p, q, r])
        return euler_rates
    
    def motor_command_to_speed(self, u_normalized):
        """Convert normalized motor commands to commanded speeds"""
        return (self.w_max - self.w_min) * (
            self.k_motor * u_normalized**2 + (1 - self.k_motor) * u_normalized
        ) + self.w_min
    
    def compute_motor_thrusts(self, motor_speeds):
        """
        Compute individual motor thrusts from motor speeds.
        
        Each motor produces thrust proportional to ω²
        T_i = k_thrust_i * ω_i²
        
        Returns: thrust vector for each motor (positive = downward in body frame)
        """
        thrusts = self.k_thrust * (motor_speeds ** 2)
        return thrusts
    
    def compute_forces_and_moments(self, v_body, motor_speeds, motor_accel):
        """
        Compute forces and moments from individual motor thrusts and positions.
        
        Args:
            v_body: velocity in body frame
            motor_speeds: angular velocities of motors
            motor_accel: angular accelerations of motors
            
        Returns:
            F_body: force in body frame
            M_body: moment in body frame
        """
        
        # THRUST from each motor (acts downward in body frame)
        thrusts = self.compute_motor_thrusts(motor_speeds)
        
        # Total thrust (sum of all motors, acts along negative z-body)
        total_thrust = np.sum(thrusts)
        F_thrust = np.array([0, 0, total_thrust])
        
        # DRAG FORCES from linear and quadratic air resistance
        vx_B, vy_B, vz_B = v_body
        
        # Linear drag (proportional to velocity)
        F_drag_linear = -self.k_drag_linear * v_body
        
        # Quadratic drag (proportional to velocity²)
        F_drag_quad = np.array([
            -self.k_drag_quad_xy * np.sign(vx_B) * vx_B**2,
            -self.k_drag_quad_xy * np.sign(vy_B) * vy_B**2,
            -self.k_drag_quad_z * np.sign(vz_B) * vz_B**2
        ])
        
        # Total force in body frame
        F_body = F_thrust + F_drag_linear + F_drag_quad
        
        # MOMENTS from thrust × position (lever arm)
        # τ_i = r_i × F_i, where r_i is motor position
        M_from_thrust = np.zeros(3)
        for i in range(self.n_motors):
            r_i = self.motor_positions[i]  # Position relative to CoM
            F_i = np.array([0, 0, thrusts[i]])  # Motor thrust (downward)
            tau_i = np.cross(r_i, F_i)
            M_from_thrust += tau_i
        
        # MOTOR REACTION TORQUES
        # Each spinning motor creates torque opposite to its spin
        # τ_reaction_i = direction_i * k_torque_i * ω_i²
        M_from_reaction = np.zeros(3)
        for i in range(self.n_motors):
            # Reaction torque along z-axis (rotor spin direction)
            tau_reaction = self.motor_directions[i] * self.k_torque[i] * (motor_speeds[i] ** 2)
            # For quad in X-config: all torques point along body z
            M_from_reaction[2] += tau_reaction
        
        # MOTOR ACCELERATION COUPLING (gyroscopic effect)
        # Rapid motor spin changes create gyroscopic moments
        # This is second-order but included for completeness
        M_from_accel = np.zeros(3)
        for i in range(self.n_motors):
            # Angular acceleration effect (smaller, proportional to ω̇)
            tau_accel = self.motor_directions[i] * 0.1 * self.k_torque[i] * motor_accel[i]
            M_from_accel[2] += tau_accel
        
        # Total moment
        M_body = M_from_thrust + M_from_reaction + M_from_accel
        
        return F_body, M_body
    
    def dynamics(self, x, u_normalized, t=0):
        """
        Compute state derivatives.
        
        Args:
            x: state vector [p(3), v(3), euler(3), body_rates(3), motor_speeds(4)]
            u_normalized: normalized motor commands [u1, u2, u3, u4]
            t: time
            
        Returns:
            x_dot: state derivatives
        """
        
        # Extract state
        p = x[0:3]
        v = x[3:6]
        euler = x[6:9]
        body_rates = x[9:12]
        motor_speeds = x[12:16]
        
        # Get rotation matrix
        R = self.rotation_matrix_zyx(euler)
        
        # Transform velocity to body frame
        v_body = R.T @ v
        
        # Motor commanded speeds
        w_commanded = self.motor_command_to_speed(u_normalized)
        
        # Motor dynamics: ω̇_i = (ω_c_i - ω_i) / τ
        motor_accel = (w_commanded - motor_speeds) / self.tau_motor
        
        # Compute forces and moments from individual motors
        F_body, M_body = self.compute_forces_and_moments(v_body, motor_speeds, motor_accel)
        
        # Linear acceleration in body frame (including gravity)
        a_body = F_body + np.array([0, 0, self.g])
        
        # Transform acceleration to inertial frame
        a_inertial = R @ a_body
        
        # Angular acceleration: Ω̇ = J^{-1}(M - Ω × JΩ)
        gyro_coupling = np.cross(body_rates, self.J @ body_rates)
        angular_accel = np.linalg.solve(self.J, M_body - gyro_coupling)
        
        # Euler angle rates
        euler_rates = self.body_rate_to_euler_rate(euler, body_rates)
        
        # Construct derivative vector
        x_dot = np.zeros(16)
        x_dot[0:3] = v
        x_dot[3:6] = a_inertial
        x_dot[6:9] = euler_rates
        x_dot[9:12] = angular_accel
        x_dot[12:16] = motor_accel
        
        return x_dot
    
    def simulate(self, x0, u_func, t_sim):
        """
        Simulate quadcopter dynamics using RK4.
        
        Args:
            x0: initial state
            u_func: function u(t) returning control input
            t_sim: time vector
            
        Returns:
            t: time vector
            x: state trajectory
        """
        def dynamics_wrapper(x, t):
            u = u_func(t)
            return self.dynamics(x, u, t)
        
        x = odeint(dynamics_wrapper, x0, t_sim)
        return t_sim, x
    
    def linearize_hover(self, hover_thrust=0.5):
        """
        Linearize around hover equilibrium.
        
        Args:
            hover_thrust: normalized thrust command for hovering
            
        Returns:
            A, B, C, D: state-space matrices
            x_eq: equilibrium state
            u_eq: equilibrium control
        """
        
        # Equilibrium state for hover
        x_eq = np.zeros(16)
        x_eq[2] = 0  # z = 0
        
        # For hovering, all motors have same speed
        u_eq = np.ones(4) * hover_thrust
        w_eq = self.motor_command_to_speed(u_eq)
        x_eq[12:16] = w_eq
        
        # Linearization using finite differences
        epsilon = 1e-6
        n_states = 16
        n_controls = 4
        n_outputs = 16
        
        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, n_controls))
        
        # Nominal dynamics
        x_nominal = self.dynamics(x_eq, u_eq)
        
        # Compute A matrix
        for i in range(n_states):
            x_pert = x_eq.copy()
            x_pert[i] += epsilon
            x_pert_dot = self.dynamics(x_pert, u_eq)
            A[:, i] = (x_pert_dot - x_nominal) / epsilon
        
        # Compute B matrix
        for i in range(n_controls):
            u_pert = u_eq.copy()
            u_pert[i] += epsilon
            x_pert_dot = self.dynamics(x_eq, u_pert)
            B[:, i] = (x_pert_dot - x_nominal) / epsilon
        
        # Output matrix (measure all states)
        C = np.eye(n_states)
        D = np.zeros((n_states, n_controls))
        
        return A, B, C, D, x_eq, u_eq
    
    def create_control_system(self, hover_thrust=0.5):
        """
        Create linearized state-space system.
        
        Returns:
            sys: control system
            x_eq: equilibrium state
            u_eq: equilibrium control
        """
        A, B, C, D, x_eq, u_eq = self.linearize_hover(hover_thrust)
        sys = ct.StateSpace(A, B, C, D)
        return sys, x_eq, u_eq
    
    def get_motor_info(self):
        """Print motor configuration information"""
        print("\n" + "="*60)
        print("MOTOR CONFIGURATION")
        print("="*60)
        print(f"Number of motors: {self.n_motors}")
        print(f"Motor positions (body frame):")
        for i, pos in enumerate(self.motor_positions):
            print(f"  Motor {i+1}: {pos} m")
        print(f"\nMotor thrust coefficients:")
        for i, k in enumerate(self.k_thrust):
            print(f"  Motor {i+1}: {k:.2e}")
        print(f"\nMotor torque coefficients:")
        for i, k in enumerate(self.k_torque):
            print(f"  Motor {i+1}: {k:.2e}")
        print(f"\nMotor spin directions:")
        for i, d in enumerate(self.motor_directions):
            direction_str = "CCW" if d > 0 else "CW"
            print(f"  Motor {i+1}: {direction_str}")
        print(f"\nMotor speed range: {self.w_min:.1f} - {self.w_max:.1f} rad/s")
        print(f"Motor time constant: {self.tau_motor:.3f} s")


if __name__ == "__main__":
    
    # Create quadcopter with independent motor model
    quad = QuadcopterIndependentMotors(params, motor_params)
    
    print("Quadcopter Independent Motor Model Initialized")
    quad.get_motor_info()
    
    # Linearization
    print("\n" + "="*60)
    print("LINEARIZATION ANALYSIS")
    print("="*60)
    
    sys, x_eq, u_eq = quad.create_control_system(hover_thrust=0.5)
    
    print(f"\nLinearized system around hover (u = {u_eq[0]:.2f}):")
    print(f"  States: 16 (p, v, euler, body_rates, motor_speeds)")
    print(f"  Inputs: 4 (motor commands)")
    print(f"  Outputs: 16 (all states)")
    
    # Compute poles
    poles = np.linalg.eigvals(sys.A)
    print(f"\nSystem eigenvalues:")
    print(f"  Unstable poles (Re > 0): {np.sum(poles.real > 0)}")
    print(f"  Marginally stable (|Re| < 0.01): {np.sum(np.abs(poles.real) < 0.01)}")
    print(f"  Stable poles (Re < -0.01): {np.sum(poles.real < -0.01)}")
    
    # Print some pole values
    print(f"\nFirst 5 eigenvalues:")
    for i, pole in enumerate(poles[:5]):
        if np.abs(pole.imag) < 1e-10:
            print(f"  λ_{i+1} = {pole.real:.4f}")
        else:
            print(f"  λ_{i+1} = {pole.real:.4f} ± j{np.abs(pole.imag):.4f}")
    
    print("\n" + "="*60)
    print("Independent motor model ready for simulation")
    print("="*60)
