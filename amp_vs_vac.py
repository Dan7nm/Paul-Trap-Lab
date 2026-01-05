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
        df = create_df("data/Changing Vac 0 Vdc/" + filename)
        df = df[(df['X pos'] > 0) & (df['Y pos'] > 0)]

        if not df.empty:
            # Calculate amplitude as half the peak-to-peak range
            # amplitude = (df['Y pos'].max() - df['Y pos'].min()) / 2
            amplitude = np.sqrt(2) * df['Y pos'].values.std()
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
        df = create_df(data_dir + "/" + path)
        y = df[Y_DATA_COL_NAME]
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
    df = pd.read_csv(file_path)
    mask = (df[TIME_COL_NAME] > 4000) & (df[TIME_COL_NAME] < 5000)
    df = df[mask]
    t = df[TIME_COL_NAME]
    x = df[col_name]
    plt.scatter(t, x, s=12)
    plt.show()

def plot_all_fft_averages():
    data_dir = vac_dir2
    avarage_all(data_dir, fft_z, bad_measurements, plot=True)
    avarage_all(data_dir, fft_r, bad_measurements, plot=True)


vac_dir1 = os.path.join('data', 'Changing Vac 0 Vdc')
vac_dir2 = os.path.join('data', 'changing Vdc 3570 Vac 1_1')
vdc_dir = os.path.join('data', "changing Vdc 3600V ac")

if __name__ == "__main__":
   plot_A_vs_Vdc()
   plot_all_fft_averages()