import os

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
bad_measurements = ["dc_0_fps_240.csv", "dc_600_fps_240.csv"]


def plot_A_vs_Vac():
    # Extract amplitude data for each voltage
    voltages = []
    amplitudes = []

    for voltage, filename in sorted(ac_voltage_to_file.items()):
        # Load and filter data (keep only positive positions)
        _, a = extract_data("data/Changing Vac 0 Vdc/" + filename, Z_DATA_COL_NAME)

        # Calculate amplitude as half the peak-to-peak range
        # amplitude = (df['Y pos'].max() - df['Y pos'].min()) / 2
        amplitude = np.sqrt(2) * a.std()
        voltages.append(voltage)
        amplitudes.append(amplitude)

    # Plot amplitude vs voltage
    plt.plot(voltages, amplitudes, 'o-')
    plt.xlabel('V_ac Amplitude (V)')
    plt.ylabel('Amplitude (pixels)')
    plt.grid(True)
    plt.savefig('amplitude_vs_vac.png')
    plt.show()

def plot_A_vs_Vdc():
    data_dir = 'data/changing Vdc 3570 Vac 1_1'
    data = []
    for path in os.listdir(data_dir):
        _, y = extract_data(data_dir + "/" + path)
        amplitude = y.values.std()
        v = int(path.split('_')[1]) # file format is dc_{v_value}_fps_240.csv
        data.append((v, amplitude))
    data.sort(key=lambda x: x[0])
    voltages, amplitudes = zip(*data)
    plt.plot(voltages, amplitudes, 'o-')
    plt.xlabel('V_dc Amplitude (V)')
    plt.ylabel('Amplitude (pixels)')
    plt.grid(True)
    plt.savefig('amplitude_vs_vdc.png')
    plt.show()

def plot_motion(file_path, col_name):
    t, x = extract_data(file_path, col_name)
    # start = 3000
    # stop = 3150
    # x = x[(t > start) & (t < stop)]
    # t = t[(t > start) & (t < stop)]
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel(col_name)
    plt.title(file_path.split(os.sep)[-1])
    plt.show()

def plot_all_fft_averages():
    data_dir = vdc_dir2
    avarage_all(data_dir, fft_z, bad_measurements, plot=True)
    avarage_all(data_dir, fft_r, bad_measurements, plot=True)


vac_dir = os.path.join('data', 'Changing Vac 0 Vdc')
vdc_dir1 = os.path.join('data', "changing Vdc 3600V ac")
vdc_dir2 = os.path.join('data', 'changing Vdc 3570 Vac 1_1')

if __name__ == "__main__":
    # folder = vdc_dir2
    # for file in os.listdir(folder):
    #     plot_motion(os.path.join(folder, file), R_DATA_COL_NAME)

    plot_motion(os.path.join(vdc_dir2, "dc_358_fps_240.csv"), Z_DATA_COL_NAME)
    plot_motion(os.path.join(vdc_dir2, "dc_358_fps_240.csv"), R_DATA_COL_NAME)
