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
	kind_arr = [kind for x in range(len(read))]
	return read, kind_arr

def get_next_sample_information(database_file="data/database.json"):
	db = {}
	with open(database_file) as raw_db:
		db = json.load(raw_db)
	name_feed, kind_feed = [], []
	for name, kind in db.items():
		name_feed.append(name)
		kind_feed.append(kind)
	i = 0
	while True:
		if i >= len(name_feed):
			i = 0
		try:
			yield (name_feed[i], kind_feed[i])
		except GeneratorExit:
			break
		i += 1

def convert_kind(kind):
	if kind == "drum":
		return [0, 1, 0, 0]
	elif kind == "guitar":
		return [0, 0, 1, 0]
	elif kind == "vocal":
		return [0, 0, 0, 1]
	else:
		return [1, 0, 0, 0]

def feed_samples(window_length=1*44100, database_file="data/database.json"):
	i = 0
	samples = []
	kind = ""
	name = ""
	while True:
		try: # yield the next one out of the current array #.reshape((1,window_length))
			try:
				shaped_sample = samples[i].reshape(2,window_length)
				shaped_sample = shaped_sample[1]
			except:
				shaped_sample = samples[i].reshape(1,window_length)
			yield (shaped_sample, convert_kind(kind))
		except GeneratorExit:
			break
		except: # ran out of current array, get a new one
			name, kind = next(get_next_sample_information(database_file=database_file))
			samples, kinds = get_sample(name, kind, window_length=window_length)
			del kinds
