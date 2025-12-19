import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

FILENAME = 'data/240hz_1800V'

data = pd.read_csv(FILENAME + '.csv')

PX_PER_MM = 75.0
FPS = 240.0  

x_px = np.float64(data['X pos'].values)
y_px = np.float64(data['Y pos'].values)

r_mm = x_px / PX_PER_MM
z_mm = y_px / PX_PER_MM

time = np.int32(data['Timepoint'].values)/ FPS

# Define simple sine function for fitting
def sine_func(t, A, omega, phi, offset):
    """Simple sine wave: A*sin(omega*t + phi) + offset"""
    return A * np.sin(omega * t + phi) + offset

# Use FFT to get better initial guesses
from scipy.fft import fft, fftfreq

N = len(time)
dt = np.mean(np.diff(time))
yf = fft(z_mm - np.mean(z_mm))
xf = fftfreq(N, dt)

# Find dominant frequency (positive frequencies only)
idx = np.argmax(np.abs(yf[1:N//2])) + 1
freq_guess = abs(xf[idx])

# Initial parameter guesses
A_guess = (np.max(z_mm) - np.min(z_mm)) / 2
offset_guess = np.mean(z_mm)
phi_guess = np.angle(yf[idx])
omega_guess = 2 * np.pi * freq_guess

print(f"Initial guesses from FFT:")
print(f"  Frequency: {freq_guess:.2f} Hz")
print(f"  Omega: {omega_guess:.2f} rad/s")
print(f"  Amplitude: {A_guess:.4f} mm")
print(f"  Phase: {phi_guess:.4f} rad")

p0 = [A_guess, omega_guess, phi_guess, offset_guess]

# Set bounds for fitting
bounds = ([0, omega_guess*0.9, -2*np.pi, offset_guess-0.1], 
          [A_guess*2, omega_guess*1.1, 2*np.pi, offset_guess+0.1])

# Perform the fit
popt, pcov = curve_fit(sine_func, time, z_mm, p0=p0, bounds=bounds, maxfev=10000)
A_fit, omega_fit, phi_fit, offset_fit = popt

# Generate fitted curve
time_fit = np.linspace(time.min(), time.max(), 1000)
z_fit = sine_func(time_fit, *popt)

# Print fit parameters
freq_fit = omega_fit / (2 * np.pi)
print(f"\nFitted Parameters:")
print(f"  Amplitude: {A_fit:.4f} mm")
print(f"  Omega: {omega_fit:.4f} rad/s")
print(f"  Frequency: {freq_fit:.4f} Hz")
print(f"  Phase: {phi_fit:.4f} rad")
print(f"  Offset: {offset_fit:.4f} mm")

# Calculate residuals
residuals = z_mm - sine_func(time, *popt)
rms_error = np.sqrt(np.mean(residuals**2))
print(f"  RMS Error: {rms_error:.6f} mm")

# Plot data and fit
plt.plot(time, z_mm, 'o', label='Data', markersize=3)
plt.plot(time_fit, z_fit, 'r-', label=f'Fit: {freq_fit:.2f} Hz', linewidth=2)
plt.xlabel('time (s)')
plt.ylabel('Z (mm)')
plt.title('Trajectory Plot')
plt.legend()
plt.grid(True)
plt.xlim(0,0.5)
plt.show()