import os
from typing import Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
R_DATA_COL_NAME = "X pos"
Z_DATA_COL_NAME = "Y pos"

TIME_COL_NAME = "Time"
PX_PER_MM = 64
FPS = 240


def extract_data(path: str, data_col, fps, calibration_ratio=None, tracking_id=None) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if tracking_id is not None:
        df = df[df['Tracking ID'] == tracking_id]
    df = df.sort_values([TIME_COL_NAME])
    df[TIME_COL_NAME] /= fps
    if calibration_ratio is not None:
        df[data_col] /= calibration_ratio
    return df[TIME_COL_NAME], df[data_col]


def fft_z(file_path, fps, plot=False, save=False, tracking_id=None):
    _, y_px = extract_data(file_path, Z_DATA_COL_NAME, tracking_id=tracking_id)

    z_mm = y_px / PX_PER_MM
    z_centered = z_mm - np.mean(z_mm)

    # Fourier Transform
    fft_vals = np.fft.fft(z_centered)
    frequencies = np.fft.fftfreq(len(z_centered), d=1 / fps)

    # Positive frequencies only
    mask = frequencies > 0
    pos_freqs = frequencies[mask]
    pos_amps = np.abs(fft_vals[mask])
    if plot:
        # --- PLOTTING SECTION DISABLED FOR BATCH PROCESSING ---
        plt.figure(figsize=(10, 6))
        plt.plot(pos_freqs, pos_amps)
        plt.title(f"z - {file_path.split(os.sep)[-1]}")
        plt.grid(True)
        if save:
            plt.savefig(file_path + '_fft_z.png', dpi=200)
        plt.show()
    # ------------------------------------------------------

    return pos_freqs, pos_amps



def fft_both_particles_r(file_path, fps, plot=False, save=False):
    pos_freqs, pos_amps_1 = fft_r(file_path, fps, plot=False, save=False, tracking_id=0)
    pos_freqs, pos_amps_2 = fft_r(file_path, fps, plot=False, save=False, tracking_id=1)
    if plot:
        # --- PLOTTING SECTION ---
        plt.figure(figsize=(10, 6))
        plt.plot(pos_freqs, pos_amps_1, label='Particle 1', alpha=0.7)
        plt.plot(pos_freqs, pos_amps_2, label='Particle 2', alpha=0.7)
        plt.title(f"r - {file_path.split(os.sep)[-1]}")
        plt.grid(True)
        plt.legend()
        if save:
            plt.savefig(file_path + '_fft_r_both_particles.png', dpi=200)
        plt.xlim(0, 5)  # Adjust this limit as needed
        plt.show()
    # ------------------------------------------------------

    return pos_freqs, pos_amps_1, pos_amps_2

def fft_r(file_path, fps, plot=False, save=False, tracking_id=None):
    _, x_px = extract_data(file_path, R_DATA_COL_NAME, fps,tracking_id=tracking_id)
    r_mm = x_px / PX_PER_MM
    r_centered = r_mm - np.mean(r_mm)

    # Fourier Transform
    fft_vals = np.fft.fft(r_centered)
    frequencies = np.fft.fftfreq(len(r_centered), d=1 / fps)

    # Positive frequencies only
    mask = frequencies > 0
    pos_freqs = frequencies[mask]
    pos_amps = np.abs(fft_vals[mask])

    if plot:
        # --- PLOTTING SECTION DISABLED FOR BATCH PROCESSING ---
        plt.figure(figsize=(10, 6))
        plt.plot(pos_freqs, pos_amps)
        plt.title(f"r - {file_path.split(os.sep)[-1]}")
        plt.grid(True)
        if save:
            plt.savefig(file_path + '_fft_r.png', dpi=200)
        plt.show()
    # ------------------------------------------------------

    return pos_freqs, pos_amps

def average_and_median_all(data_dir, fft_function, fps,bad_files: List[str], plot: bool = False,
                           tracking_id: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_interpolated_intensities = []
    common_freqs = np.linspace(0, fps / 2, 2000)

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    valid_files_count = 0

    for file in files:
        if file in bad_files:
            continue

        file_path = os.path.join(data_dir, file)
        freqs, amps = fft_function(file_path, fps, save=False, plot=plot, tracking_id=tracking_id)

        if len(freqs) == 0:
            continue

        interp_amps = np.interp(common_freqs, freqs, amps)
        all_interpolated_intensities.append(interp_amps)
        valid_files_count += 1

    # Convert to 2D array: (num_files, 2000_bins)
    all_intensities_matrix = np.array(all_interpolated_intensities)

    # 1. Calculate Mean (Sensitive to outliers)
    avg_intensity = np.mean(all_intensities_matrix, axis=0)

    # 2. Calculate Median (Robust to outliers)
    median_intensity = np.median(all_intensities_matrix, axis=0)

    # 3. Calculate Minimum (The baseline signal present in ALL files)
    min_intensity = np.min(all_intensities_matrix, axis=0)

    # Find dominant frequency based on Median
    max_idx = np.argmax(median_intensity)
    max_freq_median = common_freqs[max_idx]
    print(f"Dominant Frequency (Median): {max_freq_median:.2f} Hz")

    # Plotting
    domain = fft_function.__name__.split('_')[-1]
    plt.figure(figsize=(12, 6))

    plt.plot(common_freqs, avg_intensity, color='purple', alpha=0.5, label='Mean Spectrum')
    plt.plot(common_freqs, median_intensity, color='black', linewidth=2, label='Median Spectrum')
    plt.plot(common_freqs, min_intensity, color='green', alpha=0.6, linestyle='--', label='Min Spectrum')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Spectral Aggregation in {domain} (N={valid_files_count})')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim(0, 5)
    plt.show()

    return common_freqs, avg_intensity, median_intensity, min_intensity
