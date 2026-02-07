from typing import Any

from freq import *



DC_DIR = r"data/changing Vdc 3600V ac"
AC_DIR = r"data/Changing Vac 0 Vdc"

bad_measurements = ["dc_0_fps_240.csv", "dc_600_fps_240.csv", "51hz_2940v_track.csv"]
SEPARATED_AC_FORMAT = 3
SEPARATED_DC_FORMAT = 1

from scipy.signal import find_peaks


def get_second_peak_data(freqs, amps, min_dist_hz=0.5, prominence=0.1):
    """
    Returns the (frequency, amplitude) of ONLY the second highest peak.
    """
    # 1. Convert Hz distance to array bins
    freq_step = freqs[1] - freqs[0]
    distance_bins = int(min_dist_hz / freq_step)

    # 2. Find peaks
    # 'prominence' ensures the peak stands out from its surroundings (filters noise)
    # 'distance' ensures we don't pick a point on the slope of the main peak
    peaks_indices, _ = find_peaks(amps, distance=distance_bins, prominence=prominence)

    # 3. Handle edge case: Not enough peaks found
    if len(peaks_indices) < 2:
        print(f"Warning: Only {len(peaks_indices)} peak(s) found. Returning None.")
        return None, None

    # 4. Get amplitudes of found peaks
    found_amps = amps[peaks_indices]

    # 5. Sort by amplitude (Largest to Smallest)
    # zip pairs the index with its amplitude so we track the original location
    sorted_peaks = sorted(zip(peaks_indices, found_amps), key=lambda x: x[1], reverse=True)

    # 6. Select the SECOND item (index 1)
    second_peak_idx = sorted_peaks[1][0]

    return freqs[second_peak_idx], amps[second_peak_idx]

def _extract_v_A_pos_from_dir(data_dir, data_col, sep_index) -> tuple[Any, Any, Any]:
    data = []
    for path in os.listdir(data_dir):
        if not path.endswith('.csv'): continue
        _, y = extract_data(data_dir + os.sep + path, data_col)
        amplitude = y.values.std()
        position_average = y.values.mean()
        v = int(path.split('_')[sep_index])  # file format is dc_{v_value}_fps_240.csv
        data.append((v, amplitude, position_average))
    data.sort(key=lambda x: x[0])
    voltages, amplitudes, pos_averages = zip(*data)
    return voltages, amplitudes, pos_averages

def plot(voltages, pos_averages, xlabel, ylabel):
    plt.plot(voltages, pos_averages, 'o-', label='Position Average')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_A_vs_Vac(data_dir, data_col):
    voltages, amplitudes, pos_averages = _extract_v_A_pos_from_dir(data_dir, data_col, SEPARATED_AC_FORMAT)
    plot(voltages, pos_averages, 'V_ac Amplitude (V)', 'Position Average')
    plot(voltages, amplitudes, 'V_ac Amplitude (V)', 'Amplitudes')

def plot_A_vs_Vdc(data_dir, data_col):
    voltages, amplitudes, pos_averages = _extract_v_A_pos_from_dir(data_dir, data_col, SEPARATED_DC_FORMAT)
    plot(voltages, pos_averages, 'V_dc Amplitude (V)', 'Position Average')
    plot(voltages, amplitudes, 'V_dc Amplitude (V)', 'Amplitudes')


def plot_motion(file_path, col_name, tracking_id=None):
    t, x = extract_data(file_path, col_name, tracking_id=tracking_id)
    mask = (t > 5) & (t < 7)
    t = t[mask]
    x = x[mask]
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel(col_name)
    plt.title(file_path.split(os.sep)[-1])
    plt.show()


def plot_r_vs_z(file_path, tracking_id=None):
    # Extract data
    time, z = extract_data(file_path, Z_DATA_COL_NAME, tracking_id=tracking_id)
    time, r = extract_data(file_path, R_DATA_COL_NAME, tracking_id=tracking_id)

    plt.figure(figsize=(10, 6))

    # c=time maps the color to the time array
    # cmap='viridis' is a good default gradient (Blue=Early, Yellow=Late)
    sc = plt.scatter(z, r, c=time, cmap='viridis', s=6, alpha=0.7)

    # Add a colorbar to show the time scale
    cbar = plt.colorbar(sc)
    cbar.set_label('Time (seconds)')

    plt.xlabel('Z Position (mm)')
    plt.ylabel('R Position (mm)')
    plt.title(file_path.split(os.sep)[-1])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


def plot_r_vs_z_dual(file_path, fps,tracking_ids=[0, 1]):
    plt.figure(figsize=(10, 6))

    # Define distinct colormaps for each ID
    # Index 0 uses 'Blues', Index 1 uses 'Reds'
    colormaps = ['Blues', 'Reds']
    labels = [f'ID {tracking_ids[0]}', f'ID {tracking_ids[1]}']

    for i, tid in enumerate(tracking_ids):
        # Extract data for specific ID
        time, z = extract_data(file_path, Z_DATA_COL_NAME, fps,tracking_id=tid)
        time, r = extract_data(file_path, R_DATA_COL_NAME, fps, tracking_id=tid)
        z = z / px_per_mm_z_15_1
        r = r / px_per_mm_r_15_1


        if len(time) == 0:
            print(f"Warning: No data found for tracking_id={tid}")
            continue

        # Plot
        # We use vmin/vmax based on the specific particle's time to ensure the full gradient is used
        plt.scatter(r, z, c=time, cmap=colormaps[i], s=15,
                    label=labels[i], alpha=0.8, edgecolors='grey', linewidth=0.3)
    plot_config('R Position (mm)', 'Z Position (mm)', 'Particles Trajectories')
    plt.show()

def plot_dual_amplitude_vs_vac(data_dir, data_col):
    data = []
    for path in os.listdir(data_dir):
        if not path.endswith('.csv'): continue
        _, y1 = extract_data(data_dir + os.sep + path, data_col, tracking_id=0)
        _, y2 = extract_data(data_dir + os.sep + path, data_col, tracking_id=1)
        amplitude1 = y1.values.std()
        amplitude2 = y2.values.std()
        v = int(path.split('_')[SEPARATED_AC_FORMAT])  # file format is dc_{v_value}_fps_240.csv
        data.append((v, amplitude1, amplitude2))
    data.sort(key=lambda x: x[0])
    voltages, amplitudes1, amplitudes2 = zip(*data)

    plt.plot(voltages, amplitudes1, 'o-', label='Particle 1 Amplitude')
    plt.plot(voltages, amplitudes2, 'o-', label='Particle 2 Amplitude')
    plt.xlabel('V_ac Amplitude (V)')
    plt.ylabel('Amplitudes')
    plt.grid(True)
    plt.legend()
    plt.show()

def particle_distances(file_path, tracking_ids=(0, 1)) -> np.ndarray:
    _, z1 = extract_data(file_path, Z_DATA_COL_NAME, tracking_id=tracking_ids[0], fps=50)
    _, r1 = extract_data(file_path, R_DATA_COL_NAME, tracking_id=tracking_ids[0], fps=50)
    _, z2 = extract_data(file_path, Z_DATA_COL_NAME, tracking_id=tracking_ids[1], fps=50)
    _, r2 = extract_data(file_path, R_DATA_COL_NAME, tracking_id=tracking_ids[1], fps=50)
    z1, z2 = z1/px_per_mm_z_12_1, z2/px_per_mm_z_12_1
    r1, r2 = r1/px_per_mm_r_12_1, r2/px_per_mm_r_12_1
    # Calculate distance between particles at each time point
    distances = np.sqrt((z1 - z2)**2 + (r1 - r2)**2)
    return distances


def plot_distances_fft(file, fps=50, pad_factor=10, plot=True, x_lim=None):
    distances = particle_distances(file)
    file_name_parts = file.split(os.sep)[-1].split("_") # format: dc_{dc_value}_ac_{ac_value}_fps_{fps}.csv
    ac_value = file_name_parts[3]
    N = len(distances)

    # 2. Remove DC component
    distances = distances - np.mean(distances)

    # 3. Apply Window (Crucial for smooth peaks)
    window = np.hanning(N)
    windowed_signal = distances * window

    # 4. Zero Padding for High Resolution
    # We ask FFT to compute as if the signal was 10x longer.
    # It fills the rest with zeros, which acts as high-quality interpolation.
    n_padded = N * pad_factor

    fft_vals = np.fft.rfft(windowed_signal, n=n_padded)
    fft_freqs = np.fft.rfftfreq(n_padded, d=1 / fps)

    # 5. Find Exact Peak
    amplitude = np.abs(fft_vals)
    max_peak_freq, max_peak_value = get_second_peak_data(fft_freqs, amplitude)

    print(f"ac={ac_value} - Peak Frequency: {max_peak_freq:.2f} Hz with Amplitude: {max_peak_value:.2f} mm")

    if plot:
        plt.figure(figsize=(10, 5))
        # Plot only around the peak to see the detail
        plt.plot(fft_freqs, amplitude)
        plot_config('Frequency (Hz)', FF_LABEL, "Particles' Distance Frequency Spectrum")
        if x_lim:
            plt.xlim(x_lim)
        plt.show()
    return ac_value, max_peak_freq, max_peak_value

def plot_peak_difference_freq_vs_ac(data_dir, fps=50, plot_all=False, x_lim=None):
    ac_values = []
    peak_freq_values = []
    peak_freqs = []
    for file in os.listdir(data_dir):
        if not file.endswith('.csv'): continue
        ac_value, peak_freq, peak_value = plot_distances_fft(os.path.join(data_dir, file), fps=fps, plot=plot_all, x_lim=x_lim)
        ac_values.append(float(ac_value))
        peak_freq_values.append(peak_value)
        peak_freqs.append(peak_freq)
    plt.scatter(ac_values, peak_freq_values,)
    plot_config(r'$V_{ac}$ Amplitude (V)', 'Peak Frequency Amplitude (mm * s)', r'Peak Frequency Difference vs $V_{ac}$ Amplitude')
    plt.show()
    plt.scatter(ac_values, peak_freqs,)
    plot_config('$V_{ac}$ Amplitude (V)', 'Peak Frequency (Hz)', r'Peak Frequency Difference vs $V_{ac}$ Amplitude')
    if x_lim:
        plt.xlim(x_lim)
    plt.show()


vac_dir1 = os.path.join('data', 'Changing Vac 0 Vdc')
vac_dir2 = os.path.join("data", "psp dc 0")
vdc_dir1 = os.path.join('data', "changing Vdc 3600V ac")
vdc_dir2 = os.path.join('data', 'changing Vdc 3570 Vac 1_1')

dir_12_1 = os.path.join("data", "12_1")
dir_15_1 = os.path.join("data", "15_1")


if __name__ == "__main__":

    # for file in sorted(os.listdir(dir)):
    #     if not file.endswith('.csv'):
    #         continue
    #     file_path = os.path.join(dir, file)
    #     plot_r_vs_z_dual(file_path, 50)


    # for file in sorted(os.listdir(dir)):
    #     if not file.endswith('.csv'):
    #         continue
    #     file_path = os.path.join(dir, file)
    #     fft_both_particles_r(file_path, fps=50, plot=True, save=False)
    #plot_max_peak_vs_ac_r(dir_12_1, fps=50)



    # single particle fft
    fft_r("data/psp dc 0/dc_0_ac_2400_fps_240.csv", fps=240, plot=True, save=False)
    fft_r("data/psp dc 0/dc_0_ac_2800_fps_240.csv", fps=240, plot=True, save=False)
    fft_r("data/psp dc 0/dc_0_ac_3200_fps_240.csv", fps=240, plot=True, save=False)
    fft_r("data/psp dc 0/dc_0_ac_3590_fps_240.csv", fps=240, plot=True, save=False)

    #plot_peak_difference_freq_vs_ac(dir_12_1, fps=50, plot_all=False,)
    #plot_peak_difference_freq_vs_ac(dir_12_1, fps=50, plot_all=True,x_lim=(0, 5))

    #plot_r_vs_z_dual("data/15_1/50_fps/dc_53_ac_3560_fps_50.csv", fps=50)