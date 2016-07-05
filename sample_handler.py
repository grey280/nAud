# sample_handler: given a file name, reads in a file and returns an array of windows
import scipy.io.wavfile 		as wav
import numpy					as np
import random

def handle(sample, window_length=1*44100):
	read_in = wav.read(sample)
	samples = read_in[1]
	num_samples = len(samples)
	out_array = []
	for i in range(int(num_samples/window_length)):
		temp = samples[i*window_length:(i+1)*window_length]
		out_array.append(temp)
	return out_array
