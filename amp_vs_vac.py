from typing import Any

import matplotlib.pyplot as plt

from freq import *


# Map of voltage values to their corresponding data files
ac_voltage_to_file = {
    1800: '240hz_1800V.csv',
    2000: '240hz_2000V.csv',
    2200: '240hz_2200V.csv',
    2400: '240hz_2400V.csv',
    2700: '240hz_2700V.csv',
    3200: '240hz_3200V.csv',
    3570: '240hz_3570V.csv'
}
dc_values = [0, 30, 75, 150, 275, 300, 372, 450, 520, 600]
DC_DIR = r"data/changing Vdc 3600V ac"
AC_DIR = r"data/Changing Vac 0 Vdc"
bad_measurements = ["dc_0_fps_240.csv", "dc_600_fps_240.csv", "51hz_2940v_track.csv"]
SEPARATED_AC_FORMAT = 3
SEPARATED_DC_FORMAT = 1

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

        if len(time) == 0:
            print(f"Warning: No data found for tracking_id={tid}")
            continue

        # Plot
        # We use vmin/vmax based on the specific particle's time to ensure the full gradient is used
        plt.scatter(r, z, c=time, cmap=colormaps[i], s=15,
                    label=labels[i], alpha=0.8, edgecolors='grey', linewidth=0.3)

    plt.xlabel('R Position (mm)')
    plt.ylabel('Z Position (mm)')
    plt.title(f'{file_path.split(os.sep)[-1]} - Dual Particle Tracking')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_all_fft_averages():
    data_dir = vdc_dir2
    avarage_all(data_dir, fft_z, bad_measurements, plot=True)
    avarage_all(data_dir, fft_r, bad_measurements, plot=True)



vac_dir1 = os.path.join('data', 'Changing Vac 0 Vdc')
vac_dir2 = os.path.join("data", "psp dc 0")
vdc_dir1 = os.path.join('data', "changing Vdc 3600V ac")
vdc_dir2 = os.path.join('data', 'changing Vdc 3570 Vac 1_1')

dir_12_1 = os.path.join("data", "12_1")
dir_15_1 = os.path.join("data", "15_1")

if __name__ == "__main__":
    dir = dir_12_1
    # for file in sorted(os.listdir(dir)):
    #     if not file.endswith('.csv'):
    #         continue
    #     file_path = os.path.join(dir, file)
    #     plot_r_vs_z_dual(file_path, 50)


    for file in sorted(os.listdir(dir)):
        if not file.endswith('.csv'):
            continue
        file_path = os.path.join(dir, file)
        fft_both_particles_r(file_path, fps=50, plot=True, save=False)
