from typing import Any
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


def plot_motion(file_path, col_name):
    t, x = extract_data(file_path, col_name)
    mask = (t > 5) & (t < 7)
    t = t[mask]
    x = x[mask]
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel(col_name)
    plt.title(file_path.split(os.sep)[-1])
    plt.show()

def plot_all_fft_averages():
    data_dir = vdc_dir2
    avarage_all(data_dir, fft_z, bad_measurements, plot=True)
    avarage_all(data_dir, fft_r, bad_measurements, plot=True)


vac_dir1 = os.path.join('data', 'Changing Vac 0 Vdc')
vac_dir2 = os.path.join("data", "psp dc 0")
vdc_dir1 = os.path.join('data', "changing Vdc 3600V ac")
vdc_dir2 = os.path.join('data', 'changing Vdc 3570 Vac 1_1')



if __name__ == "__main__":
    plot_motion(r"data/two particles/dc_0_ac_3560_240fps_particle2.csv", Z_DATA_COL_NAME)
