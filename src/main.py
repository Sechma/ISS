# !/usr/bin/env python3


#Created at: 5.12.2021
#@author: xsechr00


from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk
import cmath
import IPython

def dft(array):
    dft_arr = []
    arr_len = array.size
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

data = data - np.mean(data)
data = data / np.abs(data).max()

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

frame_dft = dft(frame)
frame_dft = frame_dft[0:512]
myrange = np.arange(512)
plt.figure(figsize=(6,3))
plt.gca().set_xlabel('t [s]')
plt.gca().set_title('jeden dtft rámec')
plt.plot(myrange/fs, frame_dft, label="DFT")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

fft_frame = np.fft.fft(frame)
fft_frame = fft_frame[0:512]
myrange = np.arange(512)
plt.figure(figsize=(6,3))
plt.gca().set_xlabel('t [s]')
plt.gca().set_title('jeden fft rámec')
plt.plot(myrange/fs, fft_frame, label="FFT")
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# 4.4
f, t, sgr = spectrogram(data, fs)
sgr_log = 10 * np.log10(sgr+1e-20) 

plt.figure(figsize=(16,10))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

#4.5
# zpozorovane skrz priblizeni spektogramu
f = [1000,2000,3000,4000] #[Hz]
# f1 985-1000
# f2 1985-2000
# f3 2985-3000
# f4 3985-4000
#4.6
#source https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.02-Discrete-Fourier-Transform.html

sr = fs
ts = 1.0/fs
t = np.arange(0,2.8288125,ts)
sig_gen = 0
for freq in f:
    sig_gen += 2**8*np.cos(2*np.pi*freq*t)

plt.figure(figsize=(6,3))
plt.gca().set_title('4 cosinusovky na 32vzorcich')
plt.plot(sig_gen[:32])
plt.show()
from scipy.io.wavfile import write
write("../audio/4cos.wav", sr, sig_gen.astype(np.int16))

f, t, sgr = spectrogram(sig_gen, fs)
sgr_log = 10 * np.log10(sgr+1e-20) 

plt.figure(figsize=(8,5))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

#4.7 pasmova zadrz
from scipy import signal
from scipy.signal import butter, lfilter


def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

spectras = []
filtry = []

b1,a1 = butter_bandstop(985,1015,fs,5)
filt1 = butter_bandstop_filter(data, 985 , 1015, fs)
filtry.append(filt1)

b2,a2 = butter_bandstop(1970,2000,fs,5)
filt2 = butter_bandstop_filter(data, 1970, 2000, fs)
filtry.append(filt2)

b3,a3 = butter_bandstop(2970,3000,fs,5)
filt3 = butter_bandstop_filter(data, 2970, 3000, fs)
filtry.append(filt3)

b4,a4 = butter_bandstop(3970,4000,fs,5)
filt4 = butter_bandstop_filter(data, 3970, 4000, fs)
filtry.append(filt4)

"""
for i in filtry:
    f, t, sgr = spectrogram(i, fs)
    sgr_log = 10 * np.log10(sgr+1e-20) 

    plt.figure(figsize=(8,5))
    plt.pcolormesh(t,f,sgr_log)
    plt.gca().set_xlabel('Čas [s]')
    plt.gca().set_ylabel('Frekvence [Hz]')
    cbar = plt.colorbar()
    cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)
"""



aa = []
bb = []
aa.append(a1)
aa.append(a2) 
aa.append(a3) 
aa.append(a4)

bb.append(b1)
bb.append(b2) 
bb.append(b3) 
bb.append(b4)

N_imp = 64
for i in range(4):

    imp = [1, *np.zeros(N_imp-1)] # jednotkovy impuls
    h = lfilter(bb[i], aa[i], imp)

    plt.figure(figsize=(5,3))
    plt.stem(np.arange(N_imp), h, basefmt=' ')
    plt.gca().set_xlabel('$n$')
    plt.gca().set_title('Impulsní odezva $h[n]$')
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()

#4.8
for i in range(4):
    z, p, k = tf2zpk(bb[i], aa[i])
    plt.figure(figsize=(4,3.5))

# jednotkova kruznice
    ang = np.linspace(0, 2*np.pi,100)
    plt.plot(np.cos(ang), np.sin(ang))

    # nuly, poly
    plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='r', label='nuly')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='g', label='póly')

    plt.gca().set_xlabel('Realná složka $\mathbb{R}\{$z$\}$')
    plt.gca().set_ylabel('Imaginarní složka $\mathbb{I}\{$z$\}$')

    plt.grid(alpha=0.5, linestyle='--')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
