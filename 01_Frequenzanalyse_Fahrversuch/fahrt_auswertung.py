import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/LASEHER/bachelor/logs/adxl_20251201_102152.csv")

fs = 50.0

gx = df["g_x"].values
gy = df["g_y"].values
gz = df["g_z"].values


a_mag = np.sqrt(gx**2 + gy**2 + gz**2)

# High-Pass-Filter anwenden (2. Ordnung, 2 Hz) 
def highpass(sig, cutoff=2.0, order=2, fs=50.0):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff/nyq, btype='high')
    return filtfilt(b, a, sig)

a_mag_hp = highpass(a_mag, fs=fs)

# Welch-PSD berechnen 
f, Pxx = welch(a_mag_hp, fs=fs, nperseg=256)

plt.figure()
t = df["t_ms"].values / 1000  
plt.plot(t, a_mag_hp)
plt.xlabel("Zeit (s)")
plt.ylabel("Beschleunigung (gefiltert)")
plt.title("Fahrdaten – gefilterter Schärfe-Zeitverlauf")
plt.grid(True)

plt.figure()
plt.plot(f, Pxx)
plt.xlabel("Frequenz (Hz)")
plt.ylabel("Leistung (PSD)")
plt.title("Messungsergebnisse der Fahrt – Beschleunigungs-PSD")
plt.grid(True)

plt.show()
