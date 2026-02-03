"""
Example: Quadcopter Independent Motor Model - Hover Scenario
Simple skeleton for building on later.
"""
import numpy as np
import matplotlib.pyplot as plt
from quadcopter_sim_independent import QuadcopterIndependentMotors
from params import params, motor_params


def scenario_hover_simple():
    """Simple hover scenario with independent motor model"""
    
    # Initialize quadcopter
    quad = QuadcopterIndependentMotors(params, motor_params)
    
    # Initial state - at rest
    x0 = np.zeros(16)
    x0[2] = 0.1  # Start 10cm above ground
    
    # Hover control law
    def control_law(t):
        return np.ones(4) * 0.5  # All motors at 50% (hover)
    
    # Simulate
    t_sim = np.linspace(0, 5, 500)
    t, x_traj = quad.simulate(x0, control_law, t_sim)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Independent Motor Model - Hover', fontsize=14)
    
    # Position
    axes[0, 0].plot(t, x_traj[:, 0:3])
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend(['x', 'y', 'z'])
    axes[0, 0].grid(True)
    
    # Velocity
    axes[0, 1].plot(t, x_traj[:, 3:6])
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend(['vx', 'vy', 'vz'])
    axes[0, 1].grid(True)
    
    # Euler angles
    axes[1, 0].plot(t, np.rad2deg(x_traj[:, 6:9]))
    axes[1, 0].set_ylabel('Euler Angles (deg)')
    axes[1, 0].legend(['roll', 'pitch', 'yaw'])
    axes[1, 0].grid(True)
    
    # Motor speeds
    axes[1, 1].plot(t, x_traj[:, 12:16])
    axes[1, 1].set_ylabel('Motor Speed (rad/s)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend(['ω1', 'ω2', 'ω3', 'ω4'])
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    fig.savefig('scenario_hover_independent.png', dpi=150, bbox_inches='tight')
    print("Saved: scenario_hover_independent.png")
    
    return t, x_traj


if __name__ == "__main__":
    print("="*60)
    print("Independent Motor Model - Hover Scenario")
    print("="*60)
    
    t, x_traj = scenario_hover_simple()
    
    print(f"\nSimulation complete:")
    print(f"  Duration: {t[-1]:.1f} s")
    print(f"  Final position: [{x_traj[-1, 0]:.3f}, {x_traj[-1, 1]:.3f}, {x_traj[-1, 2]:.3f}] m")
    print(f"  Final motor speeds: {x_traj[-1, 12:16]}")