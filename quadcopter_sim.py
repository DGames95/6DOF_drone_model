"""
Quadcopter Simulation with State-Space Representation
Implements the dynamics from the system identification pipeline with control library
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import linalg
import control as ct
from pathlib import Path

# Load parameters
from params import params


class QuadcopterDynamics:
    """
    Quadcopter dynamics model with state-space representation
    
    State: x = [p, v, λ, Ω, ω]^T
        - p: position (x, y, z) in inertial frame
        - v: velocity (vx, vy, vz) in inertial frame
        - λ: Euler angles (roll, pitch, yaw)
        - Ω: body rates (p, q, r) in body frame
        - ω: motor speeds (ω1, ω2, ω3, ω4)
    
    Control: u = [u1, u2, u3, u4]^T (normalized motor commands [0, 1])
    """
    
    def __init__(self, params_dict):
        """Initialize quadcopter with parameters"""
        self.params = params_dict
        self.g = 9.81  # gravity
        
        # Extract parameters
        self.k_w = params_dict['k_w']
        self.k_x = params_dict['k_x']
        self.k_y = params_dict['k_y']
        self.k_x2 = params_dict['k_x2']
        self.k_y2 = params_dict['k_y2']
        self.k_angle = params_dict['k_angle']
        self.k_hor = params_dict['k_hor']
        self.k_v2 = params_dict['k_v2']
        
        # Roll moment coefficients (p-axis)
        self.k_p = [params_dict['k_p1'], params_dict['k_p2'], 
                    params_dict['k_p3'], params_dict['k_p4']]
        self.Jx = params_dict['Jx']
        
        # Pitch moment coefficients (q-axis)
        self.k_q = [params_dict['k_q1'], params_dict['k_q2'], 
                    params_dict['k_q3'], params_dict['k_q4']]
        self.Jy = params_dict['Jy']
        
        # Yaw moment coefficients (r-axis)
        self.k_r = [params_dict['k_r1'], params_dict['k_r2'], 
                    params_dict['k_r3'], params_dict['k_r4'],
                    params_dict['k_r5'], params_dict['k_r6'], 
                    params_dict['k_r7'], params_dict['k_r8']]
        self.Jz = params_dict['Jz']
        
        # Motor dynamics
        self.w_min = params_dict['w_min']
        self.w_max = params_dict['w_max']
        self.k = params_dict['k']  # motor command to speed coefficient
        self.tau = params_dict['tau']  # motor time constant
        
        # Center of mass offset
        self.rx = params_dict['rx']
        self.ry = params_dict['ry']
        self.rz = params_dict['rz']
        
    def rotation_matrix_zyx(self, euler):
        """Compute rotation matrix from Euler angles (ZYX convention)
        
        Args:
            euler: [roll, pitch, yaw] in radians
            
        Returns:
            R: 3x3 rotation matrix
        """
        phi, theta, psi = euler
        
        # Roll rotation (X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        
        # Pitch rotation (Y-axis)
        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        # Yaw rotation (Z-axis)
        Rz = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])
        
        # ZYX convention: R = Rz @ Ry @ Rx
        return Rz @ Ry @ Rx
    
    def euler_rate_to_body_rate(self, euler, euler_rates):
        """Convert Euler angle rates to body rates
        
        Args:
            euler: [roll, pitch, yaw]
            euler_rates: [roll_rate, pitch_rate, yaw_rate]
            
        Returns:
            body_rates: [p, q, r]
        """
        phi, theta, psi = euler
        phi_dot, theta_dot, psi_dot = euler_rates
        
        # Q matrix (Euler to body rates)
        Q = np.array([
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
            [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
        ])
        
        body_rates = np.linalg.inv(Q) @ np.array([phi_dot, theta_dot, psi_dot])
        return body_rates
    
    def body_rate_to_euler_rate(self, euler, body_rates):
        """Convert body rates to Euler angle rates
        
        Args:
            euler: [roll, pitch, yaw]
            body_rates: [p, q, r]
            
        Returns:
            euler_rates: [roll_rate, pitch_rate, yaw_rate]
        """
        phi, theta, psi = euler
        p, q, r = body_rates
        
        # Q matrix (Euler to body rates)
        Q = np.array([
            [1, 0, -np.sin(theta)],
            [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
            [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
        ])
        
        euler_rates = Q @ np.array([p, q, r])
        return euler_rates
    
    def motor_command_to_speed(self, u_normalized):
        """Convert normalized motor commands to commanded speeds
        
        Args:
            u_normalized: [u1, u2, u3, u4] in [0, 1]
            
        Returns:
            w_commanded: motor speeds in rad/s
        """
        return (self.w_max - self.w_min) * (self.k * u_normalized**2 + (1 - self.k) * u_normalized) + self.w_min
    
    def specific_force(self, v_body, motor_speeds):
        """Compute specific force (F/m) in body frame
        
        Args:
            v_body: velocity in body frame [vx_B, vy_B, vz_B]
            motor_speeds: [ω1, ω2, ω3, ω4]
            
        Returns:
            F_body: specific force in body frame
        """
        vx_B, vy_B, vz_B = v_body
        w = motor_speeds
        
        Fx = -self.k_x * vx_B * np.sum(w) - self.k_x2 * vx_B**2
        Fy = -self.k_y * vy_B * np.sum(w) - self.k_y2 * vy_B**2
        Fz = -self.k_w * np.sum(w**2)
        
        return np.array([Fx, Fy, Fz])
    
    def moment_body(self, motor_speeds, motor_accel):
        """Compute moment in body frame
        
        Args:
            motor_speeds: [ω1, ω2, ω3, ω4]
            motor_accel: [ω̇1, ω̇2, ω̇3, ω̇4]
            
        Returns:
            M_body: moment in body frame [Mx, My, Mz]
        """
        w = motor_speeds
        w_dot = motor_accel
        
        # Roll moment (p-axis)
        Mp = (-self.k_p[0] * w[0]**2 - self.k_p[1] * w[1]**2 + 
              self.k_p[2] * w[2]**2 + self.k_p[3] * w[3]**2)
        
        # Pitch moment (q-axis)
        Mq = (-self.k_q[0] * w[0]**2 + self.k_q[1] * w[1]**2 - 
              self.k_q[2] * w[2]**2 + self.k_q[3] * w[3]**2)
        
        # Yaw moment (r-axis)
        Mr = (-self.k_r[0] * w[0] + self.k_r[1] * w[1] + 
              self.k_r[2] * w[2] - self.k_r[3] * w[3] +
              -self.k_r[4] * w_dot[0] + self.k_r[5] * w_dot[1] + 
              self.k_r[6] * w_dot[2] - self.k_r[7] * w_dot[3])
        
        return np.array([Mp, Mq, Mr])
    
    def dynamics(self, x, u_normalized, t=0):
        """Compute state derivatives
        
        Args:
            x: state [p(3), v(3), euler(3), body_rates(3), motor_speeds(4)]
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
        
        # Get rotation matrix from body to inertial
        R = self.rotation_matrix_zyx(euler)
        
        # Transform velocity to body frame
        v_body = R.T @ v
        
        # Motor commanded speeds
        w_commanded = self.motor_command_to_speed(u_normalized)
        
        # Motor dynamics: ω̇i = (ωci - ω) / τ
        motor_accel = (w_commanded - motor_speeds) / self.tau
        
        # Specific force in body frame
        F_body = self.specific_force(v_body, motor_speeds)
        
        # Moment in body frame
        M_body = self.moment_body(motor_speeds, motor_accel)
        
        # Linear acceleration in body frame
        a_body = F_body + np.array([0, 0, self.g])
        
        # Transform acceleration to inertial frame
        a_inertial = R @ a_body
        
        # Angular acceleration in body frame
        # Ω̇ = J^-1 * (M - Ω × (J * Ω))
        J = np.diag([self.Jx, self.Jy, self.Jz])
        gyro_coupling = np.cross(body_rates, J @ body_rates)
        angular_accel = np.linalg.inv(J) @ (M_body - gyro_coupling)
        
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
        """Simulate quadcopter dynamics
        
        Args:
            x0: initial state
            u_func: function that returns control input u(t)
            t_sim: time vector for simulation
            
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
        """Linearize around hover equilibrium
        
        Args:
            hover_thrust: normalized thrust command for hovering
            
        Returns:
            A, B, C, D matrices for state-space representation
        """
        # Equilibrium state for hover
        x_eq = np.zeros(16)
        x_eq[2] = 0  # Start at z=0
        
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
        
        # Compute A matrix
        x_nominal = self.dynamics(x_eq, u_eq)
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
        """Create linearized state-space system for control library
        
        Args:
            hover_thrust: normalized thrust command for hovering
            
        Returns:
            sys: control system
            x_eq: equilibrium state
            u_eq: equilibrium control
        """
        A, B, C, D, x_eq, u_eq = self.linearize_hover(hover_thrust)
        sys = ct.StateSpace(A, B, C, D)
        return sys, x_eq, u_eq


def plot_bode_analysis(quad, hover_thrust=0.5):
    """Plot Bode diagrams for different input/output channels
    
    Args:
        quad: QuadcopterDynamics object
        hover_thrust: normalized thrust for linearization point
    """
    sys, x_eq, u_eq = quad.create_control_system(hover_thrust)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quadcopter Frequency Response (Bode Plots)', fontsize=14)
    
    # Plot different input/output pairs
    # u1 (motor 1) to vertical acceleration (output state 5)
    mag, phase, omega = ct.frequency_response(sys[5, 0], omega=np.logspace(0, 3, 100))
    axes[0, 0].semilogx(omega, 20*np.log10(mag))
    axes[0, 0].set_ylabel('Magnitude (dB)')
    axes[0, 0].set_title('Motor 1 → Vertical Acceleration')
    axes[0, 0].grid(True)
    
    axes[1, 0].semilogx(omega, phase * 180/np.pi)
    axes[1, 0].set_xlabel('Frequency (rad/s)')
    axes[1, 0].set_ylabel('Phase (degrees)')
    axes[1, 0].grid(True)
    
    # u1 to roll rate (output state 9)
    mag, phase, omega = ct.frequency_response(sys[9, 0], omega=np.logspace(0, 3, 100))
    axes[0, 1].semilogx(omega, 20*np.log10(mag))
    axes[0, 1].set_ylabel('Magnitude (dB)')
    axes[0, 1].set_title('Motor 1 → Roll Rate')
    axes[0, 1].grid(True)
    
    axes[1, 1].semilogx(omega, phase * 180/np.pi)
    axes[1, 1].set_xlabel('Frequency (rad/s)')
    axes[1, 1].set_ylabel('Phase (degrees)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def plot_step_response(quad, hover_thrust=0.5):
    """Plot step responses to different motor commands
    
    Args:
        quad: QuadcopterDynamics object
        hover_thrust: normalized thrust for linearization point
    """
    sys, x_eq, u_eq = quad.create_control_system(hover_thrust)
    
    # Time vector
    t = np.linspace(0, 5, 500)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Quadcopter Step Response', fontsize=14)
    
    # Motor 1 step response
    u_step = np.zeros((len(t), 4))
    u_step[:, 0] = hover_thrust + 0.1  # Step increase in motor 1
    
    t_out, y_out, _ = ct.forced_response(sys, T=t, U=u_step.T)
    
    # Plot vertical acceleration
    axes[0, 0].plot(t_out, y_out[5, :])
    axes[0, 0].set_ylabel('Vertical Acceleration (m/s²)')
    axes[0, 0].set_title('Motor 1 Step → Vertical Acceleration')
    axes[0, 0].grid(True)
    
    # Plot roll rate
    axes[0, 1].plot(t_out, y_out[9, :])
    axes[0, 1].set_ylabel('Roll Rate (rad/s)')
    axes[0, 1].set_title('Motor 1 Step → Roll Rate')
    axes[0, 1].grid(True)
    
    # Motor 1,2 differential (roll command)
    u_step2 = np.zeros((len(t), 4))
    u_step2[:, 0] = hover_thrust + 0.05
    u_step2[:, 1] = hover_thrust - 0.05
    u_step2[:, 2] = hover_thrust
    u_step2[:, 3] = hover_thrust
    
    t_out, y_out, _ = ct.forced_response(sys, T=t, U=u_step2.T)
    
    axes[1, 0].plot(t_out, y_out[6, :])  # Roll angle
    axes[1, 0].set_ylabel('Roll Angle (rad)')
    axes[1, 0].set_title('Differential Thrust → Roll')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(t_out, y_out[9, :])  # Roll rate
    axes[1, 1].set_ylabel('Roll Rate (rad/s)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Differential Thrust → Roll Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def plot_pole_zero(quad, hover_thrust=0.5):
    """Plot pole-zero map
    
    Args:
        quad: QuadcopterDynamics object
        hover_thrust: normalized thrust for linearization point
    """
    sys, x_eq, u_eq = quad.create_control_system(hover_thrust)
    
    # Compute poles and zeros
    poles = np.linalg.eigvals(sys.A)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot poles
    ax.scatter(poles.real, poles.imag, s=100, marker='x', color='red', linewidths=2, label='Poles')
    
    # Plot stability boundary
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Quadcopter Pole-Zero Map')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    return fig


if __name__ == "__main__":
    # Create quadcopter
    quad = QuadcopterDynamics(params)
    
    print("Quadcopter Dynamics Model Initialized")
    print(f"Motor range: {quad.w_min:.1f} - {quad.w_max:.1f} rad/s")
    print(f"Motor time constant: {quad.tau:.3f} s")
    print(f"Inertia - Jx: {quad.Jx:.3f}, Jy: {quad.Jy:.3f}, Jz: {quad.Jz:.3f}")
    
    # Linearization and control system analysis
    print("\nCreating linearized system around hover equilibrium...")
    sys, x_eq, u_eq = quad.create_control_system(hover_thrust=0.5)
    
    print(f"State-space system:")
    print(f"  States: 16 (p, v, euler, body_rates, motor_speeds)")
    print(f"  Inputs: 4 (motor commands)")
    print(f"  Outputs: 16 (all states)")
    
    # Compute poles
    poles = np.linalg.eigvals(sys.A)
    print(f"\nSystem eigenvalues (poles):")
    print(f"  Unstable poles (Re > 0): {np.sum(poles.real > 0)}")
    print(f"  Marginally stable poles (Re ≈ 0): {np.sum(np.abs(poles.real) < 0.01)}")
    print(f"  Stable poles (Re < 0): {np.sum(poles.real < -0.01)}")
    
    # Create plots
    print("\nGenerating analysis plots...")
    fig1 = plot_bode_analysis(quad)
    fig1.savefig('quadcopter_bode.png', dpi=150, bbox_inches='tight')
    print("  Saved: quadcopter_bode.png")
    
    fig2 = plot_step_response(quad)
    fig2.savefig('quadcopter_step_response.png', dpi=150, bbox_inches='tight')
    print("  Saved: quadcopter_step_response.png")
    
    fig3 = plot_pole_zero(quad)
    fig3.savefig('quadcopter_poles.png', dpi=150, bbox_inches='tight')
    print("  Saved: quadcopter_poles.png")
    
    plt.show()
