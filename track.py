import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILENAME = 'data/51hz_2940v_track'

data = pd.read_csv(FILENAME + '.csv', skiprows=[0, 2, 3])

PX_PER_MM = 75.0
FPS = 240.0  

x_px = np.float64(data['X'].values)
y_px = np.float64(data['Y'].values)

r_mm = x_px / PX_PER_MM
z_mm = y_px / PX_PER_MM

time = np.int32(data['Frame'].values)/ FPS

plt.plot(time, z_mm)
plt.xlabel('time (s)')
plt.ylabel('Z (mm)')
plt.title('Trajectory Plot')
plt.grid(True)
plt.show()