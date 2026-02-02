

params = {
    'k_w': 1.55e-06, 'k_x': 5.37e-05, 'k_y': 5.37e-05,
    'k_x2': 0.0041, 'k_y2' : 0.0151, 'k_angle': 3.12514, 'k_hor': 7.225, 'k_v2': 0.00,
    'k_p1': 4.99e-05, 'k_p2': 3.78e-05, 'k_p3': 4.82e-05, 'k_p4': 3.83e-05, 'Jx': -0.89,
    'k_q1': 2.05e-05, 'k_q2': 2.46e-05, 'k_q3': 2.02e-05, 'k_q4': 2.57e-05, 'Jy': 0.96,
    'k_r1': 3.38e-03, 'k_r2': 3.38e-03, 'k_r3': 3.38e-03, 'k_r4': 3.38e-03, 'k_r5': 3.24e-04, 'k_r6': 3.24e-04, 'k_r7': 3.24e-04, 'k_r8': 3.24e-04, 'Jz': -0.34,
    'w_min': 341.75, 'w_max': 3100.0, 'k': 0.5, 'tau': 0.03, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0
}

# Independent Motor Model Parameters
motor_params = {
    # Motor positions in body frame (relative to CoM)
    # X-quad configuration
    'motor_positions': [  # [x, y, z] for each motor
        [0.1, -0.1, 0.0],   # Motor 1 (front-left)
        [0.1, 0.1, 0.0],    # Motor 2 (front-right)
        [-0.1, -0.1, 0.0],  # Motor 3 (rear-left)
        [-0.1, 0.1, 0.0]    # Motor 4 (rear-right)
    ],
    
    # Individual motor thrust coefficients (kg*m/s² per rad²/s²)
    'k_thrust': [1.65e-06, 1.62e-06, 1.68e-06, 1.60e-06],  # Slight variations
    
    # Motor torque coefficients (drag/prop torque)
    'k_torque': [5.2e-08, 5.1e-08, 5.3e-08, 5.0e-08],  # Per rad²/s²
    
    # Motor rotation direction: +1 (CCW) or -1 (CW)
    'motor_directions': [1, -1, -1, 1],  # X-quad: 1,3 spin CW; 2,4 spin CCW
    
    # Motor inertia (kg*m²)
    'motor_inertia': [1.2e-05, 1.1e-05, 1.3e-05, 1.0e-05],
    
    # Linear drag force coefficients in body frame
    'k_drag_linear': 5.37e-05,  # Per m/s (same for x,y)
    
    # Quadratic drag coefficients
    'k_drag_quad_xy': 0.0041,
    'k_drag_quad_z': 0.0151,
    
    # Motor command to speed curve
    'k_motor': 0.5,  # Motor command curvature
    'tau_motor': 0.03,  # Motor time constant (s)
}



