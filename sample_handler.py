# sample_handler: given a file name, reads in a file and returns an array of windows
import scipy.io.wavfile 		as wav
import numpy					as np
import random
import json

def read_sample(sample, window_length=1*44100):
	read_in = wav.read(sample)
	samples = read_in[1]
	num_samples = len(samples)
	out_array = []
	for i in range(int(num_samples/(window_length*2))):
		try:
			temp = samples[i*window_length:(i+1)*window_length]
		except:
			pass
		out_array.append(temp)
	return out_array

def get_sample(sample, kind, window_length=1*44100):
	read = read_sample("cache/{}/{}.wav".format(kind, sample), window_length)
	kind_arr = np.full(read.shape, kind)
	return read, kind_arr

testdict = {
	"song1": "guitar",
	"song2": "voice",
	"song3": "drum",
	"song4": "other"
}
# outfile = open("test.txt", "w")
# for i in testarr:
# 	outfile.write("{}|{}\n".format(i[0], i[1]))
with open("data.txt", "w") as outfile:
	json.dump(testdict, outfile, sort_keys=True, indent=4, separators=(',',': '))

foo = []

with open("data.txt", "r") as infile:
	foo = json.load(infile)

for i in foo:
	print(i)