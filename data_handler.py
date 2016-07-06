# sample_handler: given a file name, reads in a file and returns an array of windows
import scipy.io.wavfile 		as wav
import numpy					as np
import json

window_length_default = 1*44100
database_file_default = "data/database.json"

def read_sample(sample, window_length=window_length_default):
	read_in = wav.read(sample)
	samples = read_in[1]
	num_samples = len(samples)
	print("read_sample: Number of samples: {}".format(num_samples))
	print("read_sample: Calculated number of windows: {}".format(int(num_samples/(window_length/2))))
	out_array = []
	for i in range(int(num_samples/(window_length/2))): # double, so we get the half-second overlap
		try:
			temp = samples[i*window_length:(i+1)*window_length]
		except:
			pass
		out_array.append(temp)
	return out_array

def get_sample(sample, kind, window_length=window_length_default):
	read = read_sample("cache/{}/{}.wav".format(kind, sample), window_length)
	kind_arr = [kind for x in range(len(read))]
	return read, kind_arr

def get_next_sample_information(database_file=database_file_default):
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
	out = []
	if kind == "drum":
		out= [0, 1, 0, 0]
	elif kind == "guitar":
		out= [0, 0, 1, 0]
	elif kind == "vocal":
		out= [0, 0, 0, 1]
	else:
		out= [1, 0, 0, 0]
	output = np.asarray(out).reshape(1,4)
	return output

def feed_single_samples(window_length=window_length_default, database_file=database_file_default):
	i = 0
	samples = []
	kind = ""
	name = ""
	while True:
		i += 1
		try: # yield the next one out of the current array #.reshape((1,window_length))
			try:
				shaped_sample = samples[i].reshape(2,window_length)
				shaped_sample = shaped_sample[1]
				shaped_sample = shaped_sample.reshape(1,window_length)
			except:
				shaped_sample = samples[i].reshape(1,window_length)
			yield (shaped_sample, convert_kind(kind))
		except GeneratorExit:
			break
		except: # ran out of current array, get a new one
			name, kind = next(get_next_sample_information(database_file=database_file))
			samples, kinds = get_sample(name, kind, window_length=window_length)
			del kinds

def feed_samples(window_length=window_length_default, database_file=database_file_default, samples_in_parallel=4):
	if samples_in_parallel > 4:
		samples_in_parallel = 4
	if samples_in_parallel < 1:
		samples_in_parallel = 1
	j = 0
	i = [0, 0, 0, 0]
	samples = [[], [], [], []]
	kind = ["", "", "", ""]
	name = ["", "", "", ""]
	while True:
		j = (j+1)%samples_in_parallel # select which sample group to use
		i[j] += 1
		try: # yield the next one out of the current array. .reshape((1,window_length))
			try:
				shaped_sample = samples[j][i[j]].reshape(2,window_length)
				shaped_sample = shaped_sample[1]
				shaped_sample = shaped_sample.reshape(1,window_length)
			except: 
				shaped_sample = samples[j][i[j]].reshape(1,window_length)
			yield (shaped_sample, convert_kind(kind[j]))
		except GeneratorExit:
			break
		except: # ran out of current array, get a new one
			name[j], kind[j] = next(get_next_sample_information(database_file=database_file))
			samples[j], kinds = get_sample(name[j], kind[j], window_length=window_length)
			i[j] = 0
			del kinds

def get_samples_from_file(file_to_read, window_length=window_length_default):
	i = 0
	samples = []
	samples = read_sample(file_to_read, window_length=window_length)
	print("Reading file {}, got {} samples.".format(file_to_read,len(samples)))
	out_samples = []
	for sample in samples:
		try:
			shaped_sample = samples[i].reshape(2,window_length)
			shaped_sample = shaped_sample[1]
			shaped_sample = shaped_sample.reshape(1,window_length)
		except:
			shaped_sample = samples[i].reshape(1,window_length)
		out_samples.append(shaped_sample)
	return out_samples