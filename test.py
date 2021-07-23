import scipy as sp
from compare_audio.sp_audio import diff as diff

val1, val2 = diff('AudioFiles/example--_gb_1.wav', 'AudioFiles/example--_gb_1.wav')

print(val1, val2)