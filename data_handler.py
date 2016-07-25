# sample_handler: Interface for instrument ID neural networks, providing file I/O.
import scipy.io.wavfile 		as wav
import numpy					as np
import json

window_length_default = 1*44100
database_file_default = "data/database.json"

counter = 0

def read_sample(sample, window_length=window_length_default):
	read_in = wav.read(sample)
	samples = read_in[1]
	num_samples = len(samples)
	out_array = []
	overlap_size = window_length/(2*44100)
	for i in range(int(num_samples/(window_length/overlap_size))):
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
	global counter # I hate that I need this but python calls GeneratorExit like there's no tomorrow so
	db = {}
	with open(database_file) as raw_db:
		db = json.load(raw_db)
	name_feed, kind_feed = [], []
	for name, kind in db.items():
		name_feed.append(name)
		kind_feed.append(kind)
	while True:
		counter = (counter+1)%len(name_feed)
		# print("Counter: {}".format(counter))
		try:
			# print(" name: {}, kind: {}".format(name_feed[counter],kind_feed[counter]))
			yield (name_feed[counter], kind_feed[counter])
		except GeneratorExit:
			break

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

def deconvert_kind(kind):
	out = ""
	if kind[0] == 1:
		out = "other"
	elif kind[1] == 1:
		out = "drum"
	elif kind[2] == 1:
		out = "guitar"
	elif kind[3] == 1:
		out = "vocal"
	return out

def feed_single_samples(window_length=window_length_default, database_file=database_file_default):
	i = 0
	samples = []
	kind = ""
	name = ""
	sample_generator = get_next_sample_information(database_file=database_file)
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
			name, kind = next(sample_generator)
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
			# shaped_sample = np.fft.fft(shaped_sample)
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