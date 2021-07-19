import librosa
import librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
from numpy.linalg import norm

y1, sr1 = librosa.load('AudioFiles/example--_gb_1.mp3') 
y2, sr2 = librosa.load('AudioFiles/example--_us_1.mp3')

plt.subplot(1, 2, 1)  
mfcc1 = librosa.feature.mfcc(y1,sr1)
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2) 
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

plt.show()

dist, cost, acc_cost, path = dtw(x = mfcc1.T, y = mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print("The normalized distance between the two : ",dist)