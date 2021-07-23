import librosa
import librosa.display
import matplotlib.pyplot as plt
from dtw import dtw
from numpy.linalg import norm
import sklearn.metrics.pairwise as metrics 

y1, sr1 = librosa.load('AudioFiles/example--_gb_1.mp3') 
y2, sr2 = librosa.load('AudioFiles/sample--_gb_1.mp3')
y3, sr3 = librosa.load('AudioFiles/cow--_gb_1.mp3')

print(type(y1), type(sr1))

plt.subplot(1, 3, 1)  
mfcc1 = librosa.feature.mfcc(y1,sr1)
librosa.display.specshow(mfcc1)

print(type(mfcc1))

plt.subplot(1, 3, 2) 
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

plt.subplot(1, 3, 3)
mfcc3 = librosa.feature.mfcc(y3, sr3)
librosa.display.specshow(mfcc3)

plt.show()

dist, cost, acc_cost, path = dtw(x = mfcc1.T, y = mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print("The normalized distance between the two : ",dist)

dist, cost, acc_cost, path = dtw(x = mfcc1.T, y = mfcc1.T, dist=lambda x, y: norm(x - y, ord=1))
print("The normalized distance between the two : ",dist)
