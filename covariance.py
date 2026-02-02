import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Load Data (Assumes a CSV file; sample creation included for demo)
def load_drone_data(file_path):
    return pd.read_csv(file_path)

# 2. Normalization Function
def normalize_imu(df, columns):
    """Calculates z-score normalization: (x - mean) / std"""
    norm_df = df.copy()
    for col in columns:
        mean, std = df[col].mean(), df[col].std()
        norm_df[f'{col}_norm'] = (df[col] - mean) / (std if std != 0 else 1)
    return norm_df

# 3. Calculate Total Force from Accelerations
def calculate_total_acceleration(df, accel_cols):
    """
    Calculate the magnitude of total acceleration from 3-axis IMU data.
    Formula: a_total = sqrt(ax^2 + ay^2 + az^2)
    
    This represents the total acceleration vector's magnitude acting on the drone.
    To get force: F = mass * a_total (Newton's 2nd Law: F = m*a)
    """
    df_copy = df.copy()
    ax_squared = df_copy[accel_cols[0]] ** 2
    ay_squared = df_copy[accel_cols[1]] ** 2
    az_squared = df_copy[accel_cols[2]] ** 2
    
    df_copy['total_acceleration_magnitude'] = np.sqrt(ax_squared + ay_squared + az_squared)
    return df_copy

def calculate_force(df, drone_mass_kg):
    """
    Convert total acceleration to force using F = m * a
    
    Args:
        df: DataFrame with 'total_acceleration_magnitude' column
        drone_mass_kg: Mass of the drone in kilograms
    
    Returns:
        DataFrame with 'total_force_newtons' column
    """
    df_copy = df.copy()
    df_copy['total_force_newtons'] = df_copy['total_acceleration_magnitude'] * drone_mass_kg
    return df_copy

# 4. Power Spectral Density (PSD) Analysis
def plot_psd(df, column, fs=100):
    """Computes and plots PSD to identify vibration frequencies."""
    f, Pxx_den = signal.welch(df[column], fs, nperseg=256)
    plt.semilogy(f, Pxx_den)
    plt.title(f'PSD of {column}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.grid()

# Execution
df = load_drone_data("/home/damian/dronerace/logs/Log715_2025-11-30_ASPIRE_DJ21_Grandpa/715/uart_logs/imu.csv")
accel_cols = ['imu_ax', 'imu_ay', 'imu_az']
df_norm = normalize_imu(df, accel_cols)

# Calculate total acceleration magnitude
df = calculate_total_acceleration(df, accel_cols)

print("=== IMU Acceleration Data Analysis ===")
print(f"\nAcceleration Statistics (m/s²):")
print(f"  X-axis (ax): min={df['imu_ax'].min():.3f}, max={df['imu_ax'].max():.3f}, mean={df['imu_ax'].mean():.3f}")
print(f"  Y-axis (ay): min={df['imu_ay'].min():.3f}, max={df['imu_ay'].max():.3f}, mean={df['imu_ay'].mean():.3f}")
print(f"  Z-axis (az): min={df['imu_az'].min():.3f}, max={df['imu_az'].max():.3f}, mean={df['imu_az'].mean():.3f}")
print(f"\nTotal Acceleration Magnitude (m/s²):")
print(f"  min={df['total_acceleration_magnitude'].min():.3f}, max={df['total_acceleration_magnitude'].max():.3f}, mean={df['total_acceleration_magnitude'].mean():.3f}")

# Calculate force (example with 1.5 kg drone - adjust to your actual drone mass)
drone_mass_kg = 1.5  # Replace with your actual drone mass
df = calculate_force(df, drone_mass_kg)

print(f"\n=== Force Analysis (for {drone_mass_kg} kg drone) ===")
print(f"Total Force (Newtons):")
print(f"  min={df['total_force_newtons'].min():.3f}, max={df['total_force_newtons'].max():.3f}, mean={df['total_force_newtons'].mean():.3f}")

# Comparison Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Individual accelerations
df_norm[[f'{c}_norm' for c in accel_cols]].plot(ax=axes[0, 0], title="Normalized Accelerations")
axes[0, 0].set_ylabel("Normalized Value")
axes[0, 0].grid()

# Raw accelerations
df[accel_cols].plot(ax=axes[0, 1], title="Raw Accelerations (m/s²)")
axes[0, 1].set_ylabel("Acceleration (m/s²)")
axes[0, 1].grid()

# Total acceleration magnitude
axes[1, 0].plot(df['total_acceleration_magnitude'], label='Total Acceleration Magnitude', linewidth=1)
axes[1, 0].set_title("Total Acceleration Magnitude (m/s²)")
axes[1, 0].set_ylabel("Magnitude (m/s²)")
axes[1, 0].grid()
axes[1, 0].legend()

# Total force
axes[1, 1].plot(df['total_force_newtons'], label=f'Total Force ({drone_mass_kg} kg drone)', color='red', linewidth=1)
axes[1, 1].set_title("Total Force on System (Newtons)")
axes[1, 1].set_ylabel("Force (N)")
axes[1, 1].grid()
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# PSD Plot for Z-axis (most relevant for drone lift/vibration)
plt.figure()
plot_psd(df, 'imu_az', fs=200) # Adjust fs to your actual IMU rate
plt.show()