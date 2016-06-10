import scipy.io.wavfile as wav
import numpy as np

temp = wav.read("cache/1985.Jimmy Buffet.Jimmy Buffet's Greatest Hits.Cheeseburger In Paradise.wav")
wav.write("dump.wav", temp[0], temp[1])
np.save("dump.npy", temp[1])