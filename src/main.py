# !/usr/bin/env python3


#Created at: 5.12.2021
#@author: xsechr00

from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
import cmath
import copy


def dft(array):
    dft_arr =  np.ndarray((1024))
    arr_len = array.size
    for i in range(arr_len):
        coef = 0
        for j in range(arr_len):
            coef += array[j] * cmath.exp(-2j * cmath.pi * i * j * (1 / arr_len))
        np.append(dft_arr,coef,axis=0)
    return dft_arr

#4.1
fs, data = wavfile.read("../audio/xsechr00.wav")

print("Pocet vzorku: ",data.size, "[Vzorek]")
time = data.size / fs
print("Delka signalu", time,"[s]")
print("MAX: ",data.max())
print("MIN:",data.min())    
#todo zobrazit na osu
     
#4.2
mid = 0
for i in range(0,data.size):
    mid += data[i]

mid = mid / data.size

if( data.max() >= data.min() ):
    data = data / data.max()
else:
    data = data / abs(data.min())

frame_size = 1024 
number_of_frames =  data.size//512
# arr = np.ndarray((radek,sloupec))

begin = 0
end = 1024
offset = 512
frames = np.ndarray((87,1024)) #number of frames, length frame

for i in range(0,87):
    val = data[begin:end]
    for j in range(0,1024):
        frames[i][j] = val[j]
    begin+=offset
    end+=offset
        
frame = frames[4]

myrange = np.arange(1024)
plt.figure(figsize=(6,3))
plt.gca().set_xlabel('t [s]')
plt.gca().set_title('jeden rámec')
plt.plot(myrange/fs, frame, label="signál")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
#*****************************************************
#4.3
"""
frame_dft = dft(frame)

myrange = np.arange(1024)
plt.figure(figsize=(6,3))
plt.gca().set_xlabel('t [s]')
plt.gca().set_title('jeden dtft rámec')
plt.plot(myrange/fs, frame_dft, label="DFT")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

fft_frame = np.fft.fft(frame)
myrange = np.arange(1024)
plt.figure(figsize=(6,3))
plt.gca().set_xlabel('t [s]')
plt.gca().set_title('jeden dtft rámec')
plt.plot(myrange/fs, fft_frame, label="DFT")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()"""
#**********************************************************
# 4.4
frames2 = copy.deepcopy(frames)
DFT = []
DFT_log = []

for k in range(0,87): # jednotlivé rámce
    frames2[k] = np.pad(frames2[k], (0, 1024 - len(frames2[k])), 'constant')
    N = len(frames2[k])
    kof = np.fft.fft(frames2[k])
    DFT.append(kof)
    DFT_log.append(kof)
    res_val = 10 * np.log10(np.abs(DFT[k])**2)
    DFT_log[k] = res_val
    DFT[k] = DFT[k][:512]
    DFT_log[k] = DFT_log[k][:512]


plt.figure(figsize=(12,6))
plt.imshow(np.rot90(DFT_log), extent=[0, 1, 0, fs/2], aspect='auto')
plt.gca().set_title('DFT')
plt.gca().set_xlabel('t [s]')
plt.gca().set_ylabel('f [Hz]')
b = plt.colorbar()
b.set_label('Spektralní hustota výkonu [dB]', rotation=270,labelpad=16)
plt.tight_layout()
plt.show()

