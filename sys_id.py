

"""
System Identification for Quadcopter Independent Motor Model

Load flight test data from CSV, fit model parameters to minimize prediction error,
and plot results comparing real vs simulated data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
import pickle
from pathlib import Path

from quadcopter_sim_independent import QuadcopterIndependentMotors
from params import params, motor_params


class CSVDataLoader:
    """Load and process flight test data from Betaflight CSV"""
    
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.processed = {}
        
    def quaternion_to_euler(self, qw, qx, qy, qz):
        """Convert quaternion to Euler angles (ZYX convention)"""
        phi = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        theta = np.arcsin(2*(qw*qy - qz*qx))
        psi = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        return phi, theta, psi
    
    def rotation_matrix_zyx(self, phi, theta, psi):
        """Create rotation matrix from Euler angles"""
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
    
    def extract_and_process(self, t_start=0, t_end=None):
        """Extract relevant columns and create state/input arrays"""
        
        df = self.data
        
        # Extract raw data
        t = df['timeS'].values
        
        # Time trimming
        if t_end is None:
            t_end = t[-1]
        mask = (t >= t_start) & (t <= t_end)
        
        # Time (normalize to start at 0)
        t = t[mask].copy()
        t = t - t[0]
        
        # Position
        pos_x = df['ekf_pos[0]'].values[mask]
        pos_y = df['ekf_pos[1]'].values[mask]
        pos_z = df['ekf_pos[2]'].values[mask]
        
        # Velocity
        vel_x = df['ekf_vel[0]'].values[mask]
        vel_y = df['ekf_vel[1]'].values[mask]
        vel_z = df['ekf_vel[2]'].values[mask]
        
        # Quaternion
        qw = df['ekf_quat[0]'].values[mask]
        qx = df['ekf_quat[1]'].values[mask]
        qy = df['ekf_quat[2]'].values[mask]
        qz = df['ekf_quat[3]'].values[mask]
        
        # Convert to Euler angles
        phi, theta, psi = self.quaternion_to_euler(qw, qx, qy, qz)
        
        # Body rates (gyroscope)
        p = df['gyroADC[0]'].values[mask]
        q = df['gyroADC[1]'].values[mask]
        r = df['gyroADC[2]'].values[mask]
        
        # Motor speeds (unfiltered)
        w1 = df['omegaUnfiltered[0]'].values[mask]
        w2 = df['omegaUnfiltered[1]'].values[mask]
        w3 = df['omegaUnfiltered[2]'].values[mask]
        w4 = df['omegaUnfiltered[3]'].values[mask]
        
        # Motor commands (inputs)
        u1 = df['motor[0]'].values[mask]
        u2 = df['motor[1]'].values[mask]
        u3 = df['motor[2]'].values[mask]
        u4 = df['motor[3]'].values[mask]
        
        # Accelerometers
        ax = df['accSmooth[0]'].values[mask]
        ay = df['accSmooth[1]'].values[mask]
        az = df['accSmooth[2]'].values[mask]
        
        # Construct state vector [p(3), v(3), euler(3), body_rates(3), motor_speeds(4)]
        x_measured = np.column_stack([
            pos_x, pos_y, pos_z,           # position
            vel_x, vel_y, vel_z,           # velocity
            phi, theta, psi,               # Euler angles
            p, q, r,                       # body rates
            w1, w2, w3, w4                 # motor speeds
        ])
        
        # Normalize motor commands to [0, 1]
        u_min, u_max = u1.min(), u1.max()
        u_normalized = np.column_stack([
            (u1 - u_min) / (u_max - u_min),
            (u2 - u_min) / (u_max - u_min),
            (u3 - u_min) / (u_max - u_min),
            (u4 - u_min) / (u_max - u_min)
        ])
        
        self.processed = {
            't': t,
            'x': x_measured,
            'u': u_normalized,
            'u_denorm': np.column_stack([u1, u2, u3, u4]),
            'ax': ax, 'ay': ay, 'az': az,
            'p': p, 'q': q, 'r': r,
        }
        
        return t, x_measured, u_normalized
    
    def get_processed(self):
        """Return processed data dict"""
        return self.processed


class SystemIdentificationFitter:
    """Fit quadcopter model parameters to flight data"""
    
    def __init__(self, quad, t_data, x_measured, u_data):
        self.quad = quad
        self.t_data = t_data
        self.x_measured = x_measured
        self.u_data = u_data
        self.x0 = x_measured[0]
        
    def simulate_segment(self, params_dict):
        """Simulate with given parameters"""
        # Update motor parameters
        quad_test = QuadcopterIndependentMotors(params_dict['params'], params_dict['motor_params'])
        
        # Simulate
        x_sim = odeint(
            lambda x, t, u_func: quad_test.dynamics(x, u_func(t)),
            self.x0,
            self.t_data,
            args=(self.get_u_interp,),
            full_output=False
        )
        
        return x_sim
    
    def get_u_interp(self, t):
        """Interpolate control input at time t"""
        idx = np.searchsorted(self.t_data, t)
        if idx >= len(self.u_data):
            idx = len(self.u_data) - 1
        if idx < 0:
            idx = 0
        return self.u_data[idx]
    
    def cost_function(self, param_vector, param_keys):
        """Compute fitting error"""
        # Reconstruct parameter dict from vector
        params_dict = self.pack_params(param_vector, param_keys)
        
        try:
            x_sim = self.simulate_segment(params_dict)
        except:
            return 1e10  # Simulation failed
        
        # Compute error (weighted sum of state errors)
        error = np.sum((x_sim - self.x_measured) ** 2)
        
        return error
    
    def pack_params(self, vector, keys):
        """Convert parameter vector back to dict"""
        params_copy = params.copy()
        motor_params_copy = motor_params.copy()
        
        idx = 0
        for key in keys:
            # Check which dict the parameter belongs to
            if key in params_copy:
                params_copy[key] = vector[idx]
            elif key in motor_params_copy:
                if isinstance(motor_params_copy[key], list):
                    motor_params_copy[key] = list(vector[idx:idx+len(motor_params_copy[key])])
                    idx += len(motor_params_copy[key]) - 1
                else:
                    motor_params_copy[key] = vector[idx]
            idx += 1
        
        return {'params': params_copy, 'motor_params': motor_params_copy}
    
    def fit(self, param_keys, initial_values=None):
        """Fit parameters"""
        if initial_values is None:
            initial_values = np.array([params.get(k, motor_params.get(k)) for k in param_keys])
        
        print(f"Fitting {len(param_keys)} parameters...")
        print(f"Initial parameters: {initial_values}")
        
        result = minimize(
            self.cost_function,
            initial_values,
            args=(param_keys,),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        print(f"Optimization result: {result.message}")
        print(f"Final error: {result.fun}")
        
        return self.pack_params(result.x, param_keys)


def plot_results(t_data, x_measured, x_simulated, u_data, processed_data):
    """Plot real vs simulated data"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Position
    ax = plt.subplot(3, 3, 1)
    ax.plot(t_data, x_measured[:, 0:3], 'o-', label=['x_real', 'y_real', 'z_real'], linewidth=2, markersize=3)
    ax.plot(t_data, x_simulated[:, 0:3], 's--', label=['x_sim', 'y_sim', 'z_sim'], linewidth=2, markersize=3)
    ax.set_ylabel('Position (m)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Position')
    
    # Velocity
    ax = plt.subplot(3, 3, 2)
    ax.plot(t_data, x_measured[:, 3:6], 'o-', label=['vx_real', 'vy_real', 'vz_real'], linewidth=2, markersize=3)
    ax.plot(t_data, x_simulated[:, 3:6], 's--', label=['vx_sim', 'vy_sim', 'vz_sim'], linewidth=2, markersize=3)
    ax.set_ylabel('Velocity (m/s)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Velocity')
    
    # Euler angles
    ax = plt.subplot(3, 3, 3)
    ax.plot(t_data, np.rad2deg(x_measured[:, 6:9]), 'o-', label=['roll_real', 'pitch_real', 'yaw_real'], linewidth=2, markersize=3)
    ax.plot(t_data, np.rad2deg(x_simulated[:, 6:9]), 's--', label=['roll_sim', 'pitch_sim', 'yaw_sim'], linewidth=2, markersize=3)
    ax.set_ylabel('Euler Angles (deg)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Euler Angles')
    
    # Body rates
    ax = plt.subplot(3, 3, 4)
    ax.plot(t_data, x_measured[:, 9:12], 'o-', label=['p_real', 'q_real', 'r_real'], linewidth=2, markersize=3)
    ax.plot(t_data, x_simulated[:, 9:12], 's--', label=['p_sim', 'q_sim', 'r_sim'], linewidth=2, markersize=3)
    ax.set_ylabel('Body Rates (rad/s)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Body Rates (Gyro)')
    
    # Motor speeds
    ax = plt.subplot(3, 3, 5)
    ax.plot(t_data, x_measured[:, 12:16], 'o-', label=['w1_real', 'w2_real', 'w3_real', 'w4_real'], linewidth=2, markersize=3)
    ax.plot(t_data, x_simulated[:, 12:16], 's--', label=['w1_sim', 'w2_sim', 'w3_sim', 'w4_sim'], linewidth=2, markersize=3)
    ax.set_ylabel('Motor Speed (rad/s)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Motor Speeds')
    
    # Motor inputs
    ax = plt.subplot(3, 3, 6)
    ax.plot(t_data, u_data, 'o-', linewidth=2, markersize=3)
    ax.set_ylabel('Motor Command (normalized)')
    ax.set_title('Motor Inputs')
    ax.legend(['u1', 'u2', 'u3', 'u4'])
    ax.grid(True)
    
    # Accelerometer
    ax = plt.subplot(3, 3, 7)
    ax.plot(t_data, processed_data['ax'], 'o-', label='ax', linewidth=2, markersize=3)
    ax.plot(t_data, processed_data['ay'], 's-', label='ay', linewidth=2, markersize=3)
    ax.plot(t_data, processed_data['az'], '^-', label='az', linewidth=2, markersize=3)
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Accelerometer')
    
    # Error metrics
    ax = plt.subplot(3, 3, 8)
    errors = x_measured - x_simulated
    ax.plot(t_data, errors[:, 0:3], 'o-', label=['e_pos_x', 'e_pos_y', 'e_pos_z'], linewidth=2, markersize=3)
    ax.set_ylabel('Position Error (m)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Position Error')
    
    # Error metrics (rates)
    ax = plt.subplot(3, 3, 9)
    ax.plot(t_data, errors[:, 9:12], 'o-', label=['e_p', 'e_q', 'e_r'], linewidth=2, markersize=3)
    ax.set_ylabel('Rate Error (rad/s)')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(True)
    ax.set_title('Rate Error')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    
    # Load CSV data
    csv_path = Path('/home/damian/Documents/dronerace_testing_local/btfl_002_fixed.03.si.csv')
    print(f"Loading data from {csv_path}...")
    
    loader = CSVDataLoader(csv_path)
    t_data, x_measured, u_data = loader.extract_and_process(t_start=0, t_end=None)
    processed_data = loader.get_processed()
    
    print(f"Data shape: {x_measured.shape}")
    print(f"Time range: {t_data[0]:.3f} to {t_data[-1]:.3f} s")
    print(f"Initial state:\n{x_measured[0]}")
    
    # Create quadcopter model
    quad = QuadcopterIndependentMotors(params, motor_params)
    
    # Fit parameters
    fitter = SystemIdentificationFitter(quad, t_data, x_measured, u_data)
    
    # Define which parameters to fit
    # Start with key aerodynamic parameters
    param_keys = [
        'k_thrust[0]', 'k_thrust[1]', 'k_thrust[2]', 'k_thrust[3]',
        'k_torque[0]', 'k_torque[1]', 'k_torque[2]', 'k_torque[3]',
        'k_drag_linear', 'k_drag_quad_xy', 'k_drag_quad_z',
        'tau_motor', 'Jx', 'Jy', 'Jz'
    ]
    
    # Fit (this will take a while)
    print("\n" + "="*60)
    print("STARTING PARAMETER FITTING")
    print("="*60)
    fitted_params = fitter.fit(param_keys)
    
    # Simulate with fitted parameters
    print("\nSimulating with fitted parameters...")
    quad_fitted = QuadcopterIndependentMotors(fitted_params['params'], fitted_params['motor_params'])
    x_simulated = odeint(
        lambda x, t, u_func: quad_fitted.dynamics(x, u_func(t)),
        x_measured[0],
        t_data,
        args=(fitter.get_u_interp,)
    )
    
    # Compute errors
    error_pos = np.sqrt(np.mean((x_measured[:, 0:3] - x_simulated[:, 0:3]) ** 2))
    error_vel = np.sqrt(np.mean((x_measured[:, 3:6] - x_simulated[:, 3:6]) ** 2))
    error_rate = np.sqrt(np.mean((x_measured[:, 9:12] - x_simulated[:, 9:12]) ** 2))
    
    print(f"\nFitting Results:")
    print(f"  Position RMSE: {error_pos:.6f} m")
    print(f"  Velocity RMSE: {error_vel:.6f} m/s")
    print(f"  Rate RMSE: {error_rate:.6f} rad/s")
    
    # Plot results
    fig = plot_results(t_data, x_measured, x_simulated, u_data, processed_data)
    plt.savefig('sysid_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to sysid_results.png")
    plt.show()
    
    # Save fitted parameters
    output_file = Path('/home/damian/Documents/dronerace_testing_local/fitted_params.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(fitted_params, f)
    print(f"\nFitted parameters saved to {output_file}")
    
    print("\n" + "="*60)
    print("System Identification Complete")
    print("="*60)