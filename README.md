# Quadcopter Simulation with State-Space Representation

A comprehensive quadcopter dynamics simulator using the system identification parameters from `sysid_pipeline.py`. This implementation provides both nonlinear simulation and linearized state-space models compatible with the Python `control` library.

## Features

- **Nonlinear Dynamics**: Full implementation of quadcopter equations of motion
- **State-Space Models**: Linearized systems for control design and analysis
- **Motor Dynamics**: First-order motor model with saturation
- **Rotation Matrices**: Proper ZYX Euler angle representation
- **MATLAB-like Control**: Uses `control` library for classical control analysis
- **Multiple Scenarios**: Example flight scenarios and analysis plots

## Model Description

### State Vector
```
x = [p, v, λ, Ω, ω]ᵀ  (16 states)
  - p: Position [x, y, z] in inertial frame (m)
  - v: Velocity [vx, vy, vz] in inertial frame (m/s)
  - λ: Euler angles [roll, pitch, yaw] (rad)
  - Ω: Body rates [p, q, r] in body frame (rad/s)
  - ω: Motor speeds [ω1, ω2, ω3, ω4] (rad/s)
```

### Control Input
```
u = [u1, u2, u3, u4]ᵀ ∈ [0, 1]⁴
  Normalized motor commands
```

### Dynamics

**Position and Velocity:**
```
ṗ = v
v̇ = g·e₃ + R(λ)·F
```

**Orientation:**
```
λ̇ = Q(λ)·Ω
Ω̇ = J⁻¹·(M - Ω × (J·Ω))
```

**Motor Dynamics:**
```
ω̇ᵢ = (ωcᵢ - ωᵢ) / τ
ωcᵢ = (ωmax - ωmin)·(k·uᵢ² + (1-k)·uᵢ) + ωmin
```

**Specific Force (Body Frame):**
```
Fx = -kx·vx_B·Σω - kx2·vx_B²
Fy = -ky·vy_B·Σω - ky2·vy_B²
Fz = -kw·Σω²
```

**Moment (Body Frame):**
```
Mp = -kp1·ω₁² - kp2·ω₂² + kp3·ω₃² + kp4·ω₄²
Mq = -kq1·ω₁² + kq2·ω₂² - kq3·ω₃² + kq4·ω₄²
Mr = -kr1·ω₁ + kr2·ω₂ + kr3·ω₃ - kr4·ω₄ - kr5·ω̇₁ + kr6·ω̇₂ + kr7·ω̇₃ - kr8·ω̇₄
```

**Motor Spin-Up Coupling**: The yaw moment includes terms proportional to motor angular acceleration (ω̇) representing the reaction moment from the motor's angular acceleration.

## Two Simulation Implementations

### 1. Global Coefficient Model (`quadcopter_sim.py`)
Uses aggregate body-frame coefficients for forces and moments. Suitable for hover and moderate maneuvers.

### 2. Independent Motor Model (`quadcopter_sim_independent.py`)
Each motor modeled individually with:
- **Per-motor thrust**: `T_i = k_thrust_i · ωᵢ²` 
- **Geometric moments**: `τ_i = r_i × F_i` (motor position × thrust)
- **Motor reaction torques**: Direction-specific spinning torques
- **Motor-specific coefficients**: Slight variations between motors

**Use independent model for:**
- Asymmetric motor failures/degradation
- Frame deformation analysis
- High-fidelity motor placement studies
- Aggressive aerobatic maneuvers

Both models share identical state representation and linearization approach.

## Installation

```bash
pip install numpy scipy matplotlib control
```

## Usage

### Basic Simulation

```python
from quadcopter_sim import QuadcopterDynamics
from sysid_pipeline import params

# Create quadcopter
quad = QuadcopterDynamics(params)

# Initial state (at rest)
x0 = np.zeros(16)
x0[2] = 0.5  # 0.5m altitude

# Control function
def control_law(t):
    u = np.ones(4) * 0.5  # Hover command
    if t > 2:
        u[0] += 0.05  # Roll command
    return u

# Simulate
t_sim = np.linspace(0, 10, 1000)
t, x_traj = quad.simulate(x0, control_law, t_sim)

# Plot results
plt.figure()
plt.plot(t, x_traj[:, 2])  # Altitude
plt.ylabel('Altitude (m)')
plt.xlabel('Time (s)')
plt.show()
```

### State-Space Analysis

```python
# Get linearized system around hover
sys, x_eq, u_eq = quad.create_control_system(hover_thrust=0.5)

# Frequency response
mag, phase, omega = ct.frequency_response(
    sys[5, 0],  # Motor 1 to vertical acceleration
    omega=np.logspace(0, 3, 100)
)

# Step response
t = np.linspace(0, 5, 500)
t_out, y_out = ct.step_response(sys[5, 0], T=t)

# Poles and stability
poles = np.linalg.eigvals(sys.A)
print(f"Stable poles: {np.sum(poles.real < 0)}")
print(f"Unstable poles: {np.sum(poles.real > 0)}")
```

### Run Examples

```bash
python examples.py
```

This generates:
- `scenario1_hovering.png` - Hovering with roll disturbance
- `scenario2_takeoff.png` - Takeoff from ground
- `scenario3_forward_flight.png` - Forward flight scenario
- `analysis_bode.png` - Bode magnitude and phase plots
- `analysis_step_response.png` - Step responses to motor inputs
- `analysis_poles.png` - Pole-zero map showing stability

## Class Reference

### `QuadcopterDynamics` (Global Coefficient Model)

#### Methods

**`__init__(params_dict)`**
- Initialize with parameters from system identification

**`dynamics(x, u_normalized, t=0)`**
- Compute state derivatives
- Returns: 16-element array of state derivatives

### `QuadcopterIndependentMotors` (Per-Motor Model)

#### Methods

**`__init__(params_dict, motor_params_dict)`**
- Initialize with global parameters and per-motor configuration

**`compute_motor_thrusts(motor_speeds)`**
- Calculate individual motor thrusts from speeds

**`compute_forces_and_moments(v_body, motor_speeds, motor_accel)`**
- Compute forces from lever arms and motor reaction torques
- Accounts for thrust × position and motor spin direction

**`dynamics(x, u_normalized, t=0)`**
- Compute state derivatives using independent motor model
- Returns: 16-element array of state derivatives

Both models implement identical interfaces for simulation and linearization.

**`simulate(x0, u_func, t_sim)`**
- Simulate nonlinear dynamics
- Args:
  - `x0`: Initial state (16-element array)
  - `u_func`: Function u(t) returning control input
  - `t_sim`: Time vector
- Returns: (t, x_trajectory)

**`linearize_hover(hover_thrust=0.5)`**
- Linearize around hover equilibrium
- Returns: (A, B, C, D) matrices, equilibrium state, equilibrium control

**`create_control_system(hover_thrust=0.5)`**
- Create linearized state-space system (control library)
- Returns: (sys, x_eq, u_eq)

#### Internal Methods

- `rotation_matrix_zyx(euler)` - ZYX Euler rotation matrix
- `euler_rate_to_body_rate(euler, euler_rates)` - Euler rate transformation
- `body_rate_to_euler_rate(euler, body_rates)` - Body rate transformation
- `motor_command_to_speed(u_normalized)` - Command to motor speed
- `specific_force(v_body, motor_speeds)` - Thrust and drag forces
- `moment_body(motor_speeds, motor_accel)` - Body moments/torques

## Motor Configuration

The quadcopter uses a standard X-frame configuration:

```
Motor 1 (Front-Left)      Motor 2 (Front-Right)
  ↻                         ↺
    \                       /
      \       Drone       /
      /                   \
  ↺  /                     \ ↻
Motor 3 (Rear-Left)      Motor 4 (Rear-Right)
```

**Motor Pairing for Control:**
- Roll: Motors 1,2 vs 3,4 (differential pitch moment)
- Pitch: Motors 1,3 vs 2,4 (differential roll moment)
- Yaw: Motors 1,4 vs 2,3 (differential yaw moment)
- Thrust: All motors (z-axis force)

## Parameters

Key parameters from `sysid_pipeline.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k_w` | 1.55e-06 | Thrust coefficient |
| `k_x`, `k_y` | 5.37e-05 | Linear drag coefficients |
| `k_x2`, `k_y2` | 0.0041, 0.0151 | Quadratic drag coefficients |
| `k_p1-4` | ~4.99e-05 | Roll moment coefficients |
| `k_q1-4` | ~2.05e-05 | Pitch moment coefficients |
| `k_r1-4` | ~3.38e-03 | Yaw moment coefficients (speed) |
| `k_r5-8` | ~3.24e-04 | Yaw moment coefficients (acceleration) |
| `Jx`, `Jy`, `Jz` | -0.89, 0.96, -0.34 | Moments of inertia |
| `w_min`, `w_max` | 341.75, 3100 | Motor speed limits (rad/s) |
| `tau` | 0.03 | Motor time constant (s) |

## Linearization Points

The linearized state-space model can be created around different operating points:

```python
# Hover (symmetric thrust)
sys_hover, _, _ = quad.create_control_system(hover_thrust=0.5)

# Higher altitude hover (higher thrust)
sys_high, _, _ = quad.create_control_system(hover_thrust=0.6)

# Forward flight (asymmetric thrust)
# Requires custom linearization point modification
```

## Analysis Tools

### Bode Plots
- Frequency response magnitude and phase
- Identifies control bandwidth and resonances
- Shows system gains at different frequencies

### Step Response
- Time-domain response to step inputs
- Reveals settling time and overshoot
- Useful for controller tuning

### Pole-Zero Maps
- Eigenvalue locations in complex plane
- Stability assessment (Re < 0 = stable)
- Damping ratio and natural frequency estimation

## Notes

1. **Coordinate Frames:**
   - Inertial frame: Standard NED or NWU convention (Z-up)
   - Body frame: X-forward, Y-right, Z-down
   - Motors labeled 1-4 in standard X-quad configuration

2. **Motor Spin-Up Coupling:**
   - The `kr5-8` terms model reaction torque from motor acceleration
   - This is important for rapid throttle changes

3. **Linearization:**
   - Valid for small perturbations around equilibrium
   - Hover equilibrium is the primary operating point
   - Nonlinear simulation required for large-angle maneuvers

4. **Inertia Matrix:**
   - Diagonal inertia assumed (principal axes alignment)
   - Negative values in parameters are handled in moment equations

## References

- Mellinger, D., & Kumar, V. (2011). Minimum snap trajectory generation and control for quadrotors
- Beard, R. W., & McLain, T. W. (2012). Small Unmanned Aircraft: Theory and Practice
- Mahony, R., Kumar, V., & Corke, P. (2012). Multirotor aerial vehicles

## License

Educational/Research use

## Author Notes

This simulation integrates full 6-DOF rigid body dynamics with realistic motor models estimated from flight data. The state-space linearization enables classical control analysis and controller design using standard MATLAB-style tools in Python.
