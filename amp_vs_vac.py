import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Map of voltage values to their corresponding data files
voltage_to_file = {
    1800: '240hz_1800V.csv',
    2000: '240hz_2000V.csv',
    2200: '240hz_2200V.csv',
    2400: '240hz_2400V.csv',
    2700: '240hz_2700V.csv',
    3200: '240hz_3200V.csv',
    3570: '240hz_3570V.csv'
}

# Extract amplitude data for each voltage
voltages = []
amplitudes = []

for voltage, filename in sorted(voltage_to_file.items()):
    # Load and filter data (keep only positive positions)
    df = pd.read_csv("data/" + filename)
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