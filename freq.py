import os
from typing import Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
R_DATA_COL_NAME = "X pos"
Z_DATA_COL_NAME = "Y pos"

TIME_COL_NAME = "Time"
PX_PER_MM = 75.0
FPS = 240


def extract_data(path: str, data_col, calibration_ratio=None) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df = df.sort_values([TIME_COL_NAME])
    if data_col == R_DATA_COL_NAME:
        mask = (df[data_col] > 20) & (df[data_col] < 180)
    else:
        mask = df[data_col] > 20
    df = df[mask]
    df[TIME_COL_NAME] /= FPS
    if calibration_ratio is not None:
        df[data_col] /= calibration_ratio
    return df[TIME_COL_NAME], df[data_col]


def fft_z(file_path, fps, plot=False, save=False):
    _, y_px = extract_data(file_path, Z_DATA_COL_NAME)

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


def fft_r(file_path, fps, plot=False, save=False):
    _, x_px = extract_data(file_path, R_DATA_COL_NAME)
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


def avarage_all(data_dir, fft_function, bad_files:List[str], plot:bool=False):
    all_interpolated_intensities = []

    # 1. Create a common frequency grid (0 to Nyquist)
    # Using 2000 points ensures high enough resolution for alignment
    common_freqs = np.linspace(0, FPS / 2, 2000)

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Processing {len(files)} files...")

    valid_files_count = 0
    for file in files:
        if file in bad_files:
            continue
        file_path = os.path.join(data_dir, file)

        # Get individual FFT result
        freqs, amps = fft_function(file_path, FPS, save=False, plot=plot)

        if len(freqs) == 0:
            continue

        # 2. Interpolate onto the common grid
        # This aligns the bins even if files have different lengths
        interp_amps = np.interp(common_freqs, freqs, amps)
        all_interpolated_intensities.append(interp_amps)
        valid_files_count += 1

    # 3. Average the intensities
    all_interpolated_intensities = np.array(all_interpolated_intensities)
    avg_intensity = np.mean(all_interpolated_intensities, axis=0)

    # Find dominant frequency
    max_idx = np.argmax(avg_intensity)
    max_freq = common_freqs[max_idx]

    print(f"\n--- Final Result ---")
    print(f"Dominant Frequency: {max_freq:.2f} Hz")
    domain = fft_function.__name__.split('_')[-1]
    plt.figure(figsize=(10, 6))
    plt.plot(common_freqs, avg_intensity, color='purple', label=f'Averaged Spectrum in {domain}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Average Amplitude')
    plt.title(f'Averaged FFT Spectrum in {domain} (N={valid_files_count})')
    plt.legend()
    plt.grid(True)
    plt.show()

    return common_freqs, avg_intensity