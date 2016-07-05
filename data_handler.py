# sample_handler: given a file name, reads in a file and returns an array of windows
import scipy.io.wavfile 		as wav
import numpy					as np
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

def get_next_sample_information(database_file="data/database.json"):
	with open(database_file) as raw_db:
		db = json.load(raw_db)
		samples_feed, kind_feed = [], []
		for samples, kind in db.items():
			samples_feed.append(samples)
			kind_feed.append(kind)
		i = 0
		while true:
			if i >= len(samples_feed):
				i = 0
			yield samples_feed[i], kind_feed[i]
			i += 1