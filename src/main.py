# !/usr/bin/env python3


#Created at: 5.12.2021
#@author: xsechr00

from scipy.io import wavfile
import numpy as np
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk

def dft(array):
    fill = numpy.zeros(1024-len(array))
    array = numpy.append(array, fill)
    dft_arr = []
    arr_len = len(array)
    for i in range(arr_len):
        coef = 0
        for j in range(arr_len):
            coef += array[j] * cmath.exp(-2j * cmath.pi * i * j * (1 / arr_len))
        dft_arr.append(coef)
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

# 45264 / 1024 == 45
frames = numpy.ndarray(( , ))
#4.3
