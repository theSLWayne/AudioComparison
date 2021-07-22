import numpy as np
import scipy as sp
from scipy.io.wavfile import read  
from scipy.io.wavfile import write     
from scipy import signal  
import matplotlib.pyplot as plt 

(Frequency1, array1) = read('AudioFiles/example--_gb_1.wav')
(Frequency2, array2) = read('AudioFiles/example--_us_1.wav')
(Frequency3, array3) = read('AudioFiles/sample--_gb_1.wav')

print(len(array1))
print(Frequency1)

plt.subplot(1, 3, 1)
plt.plot(array1)
plt.title('Example - British accent')  
plt.xlabel('Frequency(Hz)')  
plt.ylabel('Amplitude')

plt.subplot(1, 3, 2)
plt.plot(array2)
plt.title('Example - US accent')  
plt.xlabel('Frequency(Hz)')  
plt.ylabel('Amplitude')

plt.subplot(1, 3, 3)
plt.plot(array3)
plt.title('Sample - British accent')  
plt.xlabel('Frequency(Hz)')  
plt.ylabel('Amplitude')

plt.show()

ax1 = plt.subplot(221)
plt.plot(array1)
ax2 = plt.subplot(222)
plt.plot(array2)
plt.subplot(223, sharex = ax1)
Pxx1, freqs1, bins1, im1 = plt.specgram(array1, Fs = 1, noverlap=200)
plt.subplot(224, sharex = ax2)
Pxx2, freqs2, bins2, im2 = plt.specgram(array2, Fs = 1, noverlap=200)
plt.show()

print(Pxx1 - Pxx2)