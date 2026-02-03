"""
Example: Quadcopter Simulation Scenarios
Demonstrates different flight scenarios and control analysis

Two implementations available:
1. QuadcopterDynamics (quadcopter_sim.py)
   - Global body-frame coefficients
   - Fast computation
   - Suitable for hover and moderate maneuvers

2. QuadcopterIndependentMotors (quadcopter_sim_independent.py)
   - Per-motor models with geometric moment arms
   - Individual motor thrust/torque coefficients
   - Accurate for asymmetric configurations
   - Better for aggressive maneuvers

Both share identical state representation and control interfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys
sys.path.insert(0, '.')

from quadcopter_sim import QuadcopterDynamics, plot_bode_analysis, plot_step_response, plot_pole_zero
from params import params

def scenario_hovering():
    """Scenario 1: Hovering with step disturbance"""
    print("\n" + "="*60)
    print("SCENARIO 1: Hovering with Step Disturbance")
    print("="*60)
    
    quad = QuadcopterDynamics(params)
    
    # Initial state - at rest at origin
    x0 = np.zeros(16)
    x0[2] = 0  # z position
    
    # For hovering, all motors need equal speed to balance gravity
    hover_thrust = 0.5
    u_hover = np.ones(4) * hover_thrust
    w_hover = quad.motor_command_to_speed(u_hover)
    x0[12:16] = w_hover  # Initial motor speeds at hover equilibrium
    
    # Control function: hover for first 2s, then apply roll command
    def control_law(t):
        u = np.ones(4) * hover_thrust
        if t > 2:
            # Differential thrust for roll (motor 1,2)
            u[0] += 0.05  # increase motor 1
            u[1] -= 0.05  # decrease motor 2
        return u
    
    # Simulate
    t_sim = np.linspace(0, 6, 1000)
    t, x_traj = quad.simulate(x0, control_law, t_sim)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Scenario 1: Hovering with Roll Disturbance', fontsize=14)
    
    # Position
    axes[0, 0].plot(t, x_traj[:, 0], label='x')
    axes[0, 0].plot(t, x_traj[:, 1], label='y')
    axes[0, 0].plot(t, x_traj[:, 2], label='z')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axvline(x=2, color='r', linestyle='--', alpha=0.3)
    
    # Velocity
    axes[0, 1].plot(t, x_traj[:, 3], label='vx')
    axes[0, 1].plot(t, x_traj[:, 4], label='vy')
    axes[0, 1].plot(t, x_traj[:, 5], label='vz')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axvline(x=2, color='r', linestyle='--', alpha=0.3)
    
    # Euler angles
    axes[1, 0].plot(t, np.rad2deg(x_traj[:, 6]), label='roll')
    axes[1, 0].plot(t, np.rad2deg(x_traj[:, 7]), label='pitch')
    axes[1, 0].plot(t, np.rad2deg(x_traj[:, 8]), label='yaw')
    axes[1, 0].set_ylabel('Euler Angles (deg)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].axvline(x=2, color='r', linestyle='--', alpha=0.3)
    
    # Body rates
    axes[1, 1].plot(t, np.rad2deg(x_traj[:, 9]), label='p (roll rate)')
    axes[1, 1].plot(t, np.rad2deg(x_traj[:, 10]), label='q (pitch rate)')
    axes[1, 1].plot(t, np.rad2deg(x_traj[:, 11]), label='r (yaw rate)')
    axes[1, 1].set_ylabel('Body Rates (deg/s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axvline(x=2, color='r', linestyle='--', alpha=0.3)
    
    # Motor speeds
    axes[2, 0].plot(t, x_traj[:, 12], label='ω1')
    axes[2, 0].plot(t, x_traj[:, 13], label='ω2')
    axes[2, 0].plot(t, x_traj[:, 14], label='ω3')
    axes[2, 0].plot(t, x_traj[:, 15], label='ω4')
    axes[2, 0].set_ylabel('Motor Speed (rad/s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    axes[2, 0].axvline(x=2, color='r', linestyle='--', alpha=0.3)
    
    # Motor commands
    u_traj = np.array([control_law(ti) for ti in t])
    axes[2, 1].plot(t, u_traj[:, 0], label='u1')
    axes[2, 1].plot(t, u_traj[:, 1], label='u2')
    axes[2, 1].plot(t, u_traj[:, 2], label='u3')
    axes[2, 1].plot(t, u_traj[:, 3], label='u4')
    axes[2, 1].set_ylabel('Motor Command (normalized)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    axes[2, 1].axvline(x=2, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('scenario1_hovering.png', dpi=150, bbox_inches='tight')
    print("Saved: scenario1_hovering.png")
    
    return fig


def scenario_takeoff():
    """Scenario 2: Takeoff from ground"""
    print("\n" + "="*60)
    print("SCENARIO 2: Takeoff from Ground")
    print("="*60)
    
    quad = QuadcopterDynamics(params)
    
    # Initial state - on ground at rest
    x0 = np.zeros(16)
    x0[2] = 0.01  # Slightly above ground
    
    # Control function: ramp up thrust over 2 seconds
    def control_law(t):
        if t < 2:
            u_cmd = 0.5 + 0.2 * (t / 2)  # Ramp from 0.5 to 0.7
        else:
            u_cmd = 0.7  # Hold at 0.7 for hover
        return np.array([u_cmd, u_cmd, u_cmd, u_cmd])
    
    # Simulate
    t_sim = np.linspace(0, 8, 1000)
    t, x_traj = quad.simulate(x0, control_law, t_sim)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Scenario 2: Takeoff from Ground', fontsize=14)
    
    # Altitude and vertical velocity
    axes[0, 0].plot(t, x_traj[:, 2], linewidth=2)
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude during Takeoff')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.1)
    
    axes[0, 1].plot(t, x_traj[:, 5], linewidth=2, color='orange')
    axes[0, 1].set_ylabel('Vertical Velocity (m/s)')
    axes[0, 1].set_title('Vertical Velocity')
    axes[0, 1].grid(True)
    
    # Motor speeds
    axes[1, 0].plot(t, x_traj[:, 12:16])
    axes[1, 0].set_ylabel('Motor Speed (rad/s)')
    axes[1, 0].set_title('Motor Speeds')
    axes[1, 0].legend(['ω1', 'ω2', 'ω3', 'ω4'])
    axes[1, 0].grid(True)
    
    # Attitude (should remain near zero during symmetric thrust)
    axes[1, 1].plot(t, np.rad2deg(x_traj[:, 6:9]))
    axes[1, 1].set_ylabel('Angle (deg)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Attitude Angles')
    axes[1, 1].legend(['roll', 'pitch', 'yaw'])
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    fig.savefig('scenario2_takeoff.png', dpi=150, bbox_inches='tight')
    print("Saved: scenario2_takeoff.png")
    
    return fig


def scenario_forward_flight():
    """Scenario 3: Forward flight with pitch command"""
    print("\n" + "="*60)
    print("SCENARIO 3: Forward Flight with Pitch Command")
    print("="*60)
    
    quad = QuadcopterDynamics(params)
    
    # Initial state
    x0 = np.zeros(16)
    x0[2] = 1.0  # Start at 1m altitude
    u_hover = np.ones(4) * 0.5
    w_hover = quad.motor_command_to_speed(u_hover)
    x0[12:16] = w_hover
    
    # Control function: forward flight
    def control_law(t):
        u = np.ones(4) * 0.5  # Base hover command
        if t > 1:
            # Pitch forward by reducing motors 1,2 and increasing 3,4
            u[0] -= 0.05
            u[1] -= 0.05
            u[2] += 0.05
            u[3] += 0.05
        return u
    
    # Simulate
    t_sim = np.linspace(0, 8, 1000)
    t, x_traj = quad.simulate(x0, control_law, t_sim)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Scenario 3: Forward Flight', fontsize=14)
    
    # Position in XY plane
    axes[0, 0].plot(x_traj[:, 0], x_traj[:, 1], linewidth=2)
    axes[0, 0].scatter(x_traj[0, 0], x_traj[0, 1], color='green', s=100, label='Start', zorder=5)
    axes[0, 0].scatter(x_traj[-1, 0], x_traj[-1, 1], color='red', s=100, label='End', zorder=5)
    axes[0, 0].set_xlabel('X Position (m)')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('Flight Path (XY Plane)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    # Altitude over time
    axes[0, 1].plot(t, x_traj[:, 2], linewidth=2)
    axes[0, 1].set_ylabel('Altitude (m)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_title('Altitude')
    axes[0, 1].grid(True)
    
    # Velocity components
    axes[1, 0].plot(t, x_traj[:, 3], label='vx')
    axes[1, 0].plot(t, x_traj[:, 4], label='vy')
    axes[1, 0].plot(t, x_traj[:, 5], label='vz')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Velocity Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Attitude
    axes[1, 1].plot(t, np.rad2deg(x_traj[:, 6]), label='roll')
    axes[1, 1].plot(t, np.rad2deg(x_traj[:, 7]), label='pitch')
    axes[1, 1].plot(t, np.rad2deg(x_traj[:, 8]), label='yaw')
    axes[1, 1].set_ylabel('Angle (deg)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Attitude Angles')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    fig.savefig('scenario3_forward_flight.png', dpi=150, bbox_inches='tight')
    print("Saved: scenario3_forward_flight.png")
    
    return fig


def run_all_scenarios():
    """Run all example scenarios"""
    print("\n" + "="*60)
    print("QUADCOPTER SIMULATION - EXAMPLE SCENARIOS")
    print("="*60)
    
    try:
        fig1 = scenario_hovering()
        fig2 = scenario_takeoff()
        fig3 = scenario_forward_flight()
        
        print("\n" + "="*60)
        print("FREQUENCY DOMAIN ANALYSIS")
        print("="*60)
        
        quad = QuadcopterDynamics(params)
        
        print("\nGenerating Bode plots...")
        fig4 = plot_bode_analysis(quad)
        fig4.savefig('analysis_bode.png', dpi=150, bbox_inches='tight')
        print("Saved: analysis_bode.png")
        
        print("\nGenerating step response plots...")
        fig5 = plot_step_response(quad)
        fig5.savefig('analysis_step_response.png', dpi=150, bbox_inches='tight')
        print("Saved: analysis_step_response.png")
        
        print("\nGenerating pole-zero map...")
        fig6 = plot_pole_zero(quad)
        fig6.savefig('analysis_poles.png', dpi=150, bbox_inches='tight')
        print("Saved: analysis_poles.png")
        
        print("\n" + "="*60)
        print("ALL SCENARIOS COMPLETED SUCCESSFULLY")
        print("="*60)
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_scenarios()
