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

# Define sine function with second harmonic for fitting
def sine_func(t, A1, freq, phase1, A2, phase2, offset):
    """Sine wave with fundamental and second harmonic"""
    return (A1 * np.sin(2 * np.pi * freq * t + phase1) + 
            A2 * np.sin(2 * np.pi * 2*freq * t + phase2) + offset)

# Use FFT to get better initial guesses
from scipy.fft import fft, fftfreq

N = len(time)
dt = np.mean(np.diff(time))
yf = fft(z_mm - np.mean(z_mm))
xf = fftfreq(N, dt)

# Find dominant frequency (positive frequencies only)
idx = np.argmax(np.abs(yf[1:N//2])) + 1
freq_guess = abs(xf[idx])

# Find second harmonic amplitude
idx2 = np.argmin(np.abs(xf - 2*freq_guess))
A2_guess = 2 * np.abs(yf[idx2]) / N
phase2_guess = np.angle(yf[idx2])

# Initial parameter guesses
A_guess = (np.max(z_mm) - np.min(z_mm)) / 2
offset_guess = np.mean(z_mm)
phase_guess = np.angle(yf[idx])

print(f"Initial guesses from FFT:")
print(f"  Frequency: {freq_guess:.2f} Hz")
print(f"  Amplitude 1: {A_guess:.4f} mm")
print(f"  Amplitude 2: {A2_guess:.4f} mm")
print(f"  Phase 1: {phase_guess:.4f} rad")

p0 = [A_guess, freq_guess, phase_guess, A2_guess, phase2_guess, offset_guess]

# Set bounds for fitting
bounds = ([0, freq_guess*0.9, -2*np.pi, -A_guess, -2*np.pi, offset_guess-0.1], 
          [A_guess*2, freq_guess*1.1, 2*np.pi, A_guess, 2*np.pi, offset_guess+0.1])

# Perform the fit
popt, pcov = curve_fit(sine_func, time, z_mm, p0=p0, bounds=bounds, maxfev=10000)
A1_fit, freq_fit, phase1_fit, A2_fit, phase2_fit, offset_fit = popt

# Generate fitted curve
time_fit = np.linspace(time.min(), time.max(), 1000)
z_fit = sine_func(time_fit, *popt)

# Print fit parameters
print(f"\nFitted Parameters:")
print(f"  Amplitude (fundamental): {A1_fit:.4f} mm")
print(f"  Amplitude (2nd harmonic): {A2_fit:.4f} mm")
print(f"  Frequency: {freq_fit:.4f} Hz")
print(f"  Phase 1: {phase1_fit:.4f} rad")
print(f"  Phase 2: {phase2_fit:.4f} rad")
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