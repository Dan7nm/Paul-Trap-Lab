import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILENAME = 'data/240hz_1800V'

data = pd.read_csv(FILENAME + '.csv')

PX_PER_MM = 75.0
FPS = 240.0  

x_px = data['X pos'].values
y_px = data['Y pos'].values

r_mm = x_px / PX_PER_MM
z_mm = y_px / PX_PER_MM

time = data['Timepoint'].values / FPS

r_centered = r_mm - np.mean(r_mm)
z_centered = z_mm - np.mean(z_mm)

# Fourier Transform on z_centered
fft_z = np.fft.fft(z_centered)
frequencies = np.fft.fftfreq(len(z_centered), d=1/FPS)

# Get positive frequencies only
positive_freq_mask = frequencies > 0
positive_frequencies = frequencies[positive_freq_mask]
positive_fft_z = np.abs(fft_z[positive_freq_mask])

# Find and print the frequency with maximum amplitude
max_amplitude_idx = np.argmax(positive_fft_z)
max_frequency = positive_frequencies[max_amplitude_idx]
print(f"z-axis frequency with maximum amplitude: {max_frequency:.2f} Hz")

# Plot the frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies, positive_fft_z)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier Transform of z_centered')
plt.grid(True)
plt.savefig(FILENAME + '_fft_z.png',dpi=200)
# plt.show()

# Fourier Transform on r_centered
fft_r = np.fft.fft(r_centered)
frequencies_r = np.fft.fftfreq(len(r_centered), d=1/FPS)

# Get positive frequencies only
positive_freq_mask_r = frequencies_r > 0
positive_frequencies_r = frequencies_r[positive_freq_mask_r]
positive_fft_r = np.abs(fft_r[positive_freq_mask_r])

# Find and print the frequency with maximum amplitude
max_amplitude_idx_r = np.argmax(positive_fft_r)
max_frequency_r = positive_frequencies_r[max_amplitude_idx_r]
print(f"r-axis frequency with maximum amplitude: {max_frequency_r:.2f} Hz")

# Plot the frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(positive_frequencies_r, positive_fft_r)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier Transform of r_centered')
plt.grid(True)
plt.savefig(FILENAME + '_fft_r.png',dpi=200)
# plt.show()

