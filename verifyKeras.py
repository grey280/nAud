from keras.models 		import Sequential, model_from_json
from keras.layers 		import Dense, Dropout, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping, LearningRateScheduler

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav
import random
import time

import gdebug
import gconvert			as conv

# Settings
## Debug Settings
log_level = 2 							# 0: silent, 1: errors only, 2: normal, 3: verbose

## Neural Network settings
batch_size = 16
epoch_count = 50
data_point_count = 0 					# number of data points to use for training; set to 0 for 'all'

## IO settings
input_data = "cache/data.plist"
weights_file_name = ".json"		# name of model file to load
model_file_name = ".hdf5"		# name of weights file to load
test_series_name = "nDS"			# name of the test series - files are saved as test_series_name.iteration.[json/hdf5]
tests_in_series = 3 					# number of tests to run in this series
vstack_split_size = 35					# controls the speed/memory usage of loading tracks. 25-50 works well.
start_point = 60 						# seconds into the sample to read ((start_point+sample_duration)<sample length)
sample_duration = 15					# seconds of sample to read ((start_point+sample_duration)<sample length)
do_random_parse = True					# true will use three 5-second clips from random places in the song, rather than a single 15-second block

## Operational settings
load_from_previous_trial = True
trial_iteration_to_load = 1
trial_to_load = 0

# Tools
d = gdebug.Debugger(debug_level = log_level)

# Helper functions

def parse_track(track, data):
	# Handles track parsing - given the track and data, does whatever conversions and loading are necessary
	if do_random_parse:
		return random_parse_track(track, data)
	
	genre_orig = data.get("genre", "Unknown")
	genre = int(conv.convert_genre(genre_orig))
	scaled_genre = conv.scale_genre(genre)

	# Process sample
	sample_data = wav.read(track)
	# d.verbose("    Samples: {}".format(len(sample_data[1])))
	data = np.ndarray.flatten(sample_data[1])
	del sample_data
	start_point_calc = start_point*44100
	end_point_calc = (start_point+sample_duration)*44100
	return scaled_genre, data[start_point_calc:end_point_calc] # force it to be that size, so the NN doesn't complain

def random_parse_track(track, data):
	# Handles track parsing for the '3 5-second chunks from anywhere in the song' test group.

	genre_orig = data.get("genre", "Unknown")
	genre = int(conv.convert_genre(genre_orig))
	scaled_genre = conv.scale_genre(genre)

	# Process sample
	sample_data = wav.read(track)
	d.verbose("    Samples: {}".format(len(sample_data[1])))
	total_samples = len(sample_data[1])
	data = np.ndarray.flatten(sample_data[1])
	del sample_data
	duration = ((sample_duration/3)*44100)
	if duration >= total_samples:
		raise ValueError('Song is not long enough.')
	start_point_1 = int(random.randrange(total_samples - duration))
	start_point_2 = int(random.randrange(total_samples - duration))
	start_point_3 = int(random.randrange(total_samples - duration))
	data_1 = data[start_point_1:int(start_point_1 + duration)]
	data_2 = data[start_point_2:int(start_point_2 + duration)]
	data_3 = data[start_point_3:int(start_point_3 + duration)]
	return scaled_genre, np.concatenate((data_1, data_2, data_3))

def load_model(iteration=0, path=test_series_name):
	if load_from_previous_trial:
		load_path = "output/{}.{}.{}.json".format(path, trial_to_load, iteration)
	else:
		load_path = "output/{}".format(model_file_name)
	model = open(load_path, 'r').read()
	return model_from_json(model)

def load_weights(iteration=0, path=test_series_name):
	global model
	if load_from_previous_trial:
		load_path = "output/{}.{}.{}.hdf5".format(path, trial_to_load, iteration)
	else:
		load_path = "output/{}.hdf5".format(path)
	model.load_weights(load_path)

class Dataset:
	# Having a class to handle the dataset makes a lot of things easier.
	# Basically, hand it something opened by plistlib and it'll parse it out nice and pretty-like.
	start = 0
	def __init__(self, inpt):
		self.input_values = inpt
		self.locations=[]
		for track, data in inpt.items():
			self.locations.append(track)
		# d.debug("Initializing data set object")
	def shuffle(self):
		# Shuffles the dataset. May be expanded in the future to better handle 'holding out test data' functionality.
		random.shuffle(self.locations)
		self.start = 0

	def get_data_point_count(self):
		# This function may get expanded in the future to better handle 'holding out test data' functionality.
		return len(self.locations)

	def next_batch(self, data_point_count):
		# Loads the next batch - with optimizations, this can actually handle batch sizes in the (0,2000) range
		# pretty well - don't actually know how big it gets without trouble, that's all I've tested.
		# Of course, with big batches, loading *does* get slow, but there's not much you can do about that.
		location = self.locations[self.start]
		data_point = self.input_values.get(location)
		genre, output = parse_track(location, data_point)
		answer_feed = [genre]
		information_feed = [location]
		try:
			output = output.asarray()
		except:
			pass
		data_feed_holder = output
		data_feed = np.empty((44100*sample_duration,),dtype='int16')
		for i in range(1, data_point_count):
			if(self.start + 2 >= len(self.locations)):
				self.shuffle()
			if(i%vstack_split_size == 0):
				data_feed = np.vstack((data_feed, data_feed_holder))
				d.verbose(data_feed_holder.shape)
				del data_feed_holder
			location = self.locations[self.start]
			self.start += 1
			data_point = self.input_values.get(location)
			try:
				genre, output = parse_track(location, data_point)
			except ValueError:
				continue
			information_feed.append(location)
			d.vprogress("Loading tracks".format(location),i+1,data_point_count)
			if(i%vstack_split_size==0): # fixes an off-by-vstack_split_size error, because np.empty is *weird*
				data_feed_holder = output
			else:
				data_feed_holder = np.vstack((data_feed_holder,output))
			answer_feed.append(genre)
		data_feed = np.vstack((data_feed,data_feed_holder))
		data_array_feed = np.asarray(data_feed)[1:] # fixes an off-by-one error that you get from the way np.empty works
		answer_array_feed = np.asarray(answer_feed)
		return data_array_feed, answer_array_feed, information_feed


# Import data
# d.debug("Start: read plist")
tracks = plistlib.readPlist(input_data)
# d.debug("End: read plist")
data_set = Dataset(tracks)
# d.debug("Dataset built.")
# d.verbose("Dataset size: {}".format(data_set.get_data_point_count()))

# Load configuration, if necessary
if data_point_count == 0:
	data_point_count = data_set.get_data_point_count()

model = load_model(iteration=trial_iteration_to_load)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# d.debug("Model loaded and SGD prepared.")
load_weights(iteration=trial_iteration_to_load)
# d.debug("Weights loaded.")
d.debug("song identifier/likelihood_art/likelihood_pop/likelihood_tradition/likely_type/was_art/was_pop/was_tradition/was_type")
data_array_feed, answer_array_feed, information_feed = data_set.next_batch(data_point_count)
for i in range(len(data_array_feed)):
	# Prep for analysis
	d1 = []
	d1.append(data_array_feed[i])
	outer_data = np.asarray(d1)

	# Run analysis
	result = model.predict(outer_data, batch_size=1, verbose=0)

	# Prep result for printing
	# one_hot = result[0]
	as_int = conv.one_hot_to_int(result[0])
	# as_label = conv.number_to_label(as_int)

	# Prep 'correct' for printing
	orig_as_int = conv.one_hot_to_int(answer_array_feed[i])
	# orig_as_label = conv.number_to_label(orig_as_int)

	# Print
	# d.debug("Orig int: {} Orig label: {}".format(orig_as_int, orig_as_label))
	# d.debug("{}: Guess: {} Correct: {}".format(information_feed[i], as_label, orig_as_label))
	# d.debug("  Guess: {}  Correct: {}".format(result[0], answer_array_feed[i]))
	d.debug("{}/{}/{}/{}/{}/{}/{}/{}/{}".format(information_feed[i],result[0][0],result[0][1],result[0][2],as_int,answer_array_feed[0][0],answer_array_feed[0][1],answer_array_feed[0][2],orig_as_int))

	# Output: song_identifier, likelihood_art, likelihood_pop, likelihood_tradition, was_art, was_pop, was_tradition


# specific_song_to_test = "cache/2016.Ten Fé.NOON  189.Elodie.wav"
# that_data = {"genre": "Indie"}

# d1 = []
# scaled_genre, data = parse_track(specific_song_to_test, that_data)
# print("scaled_genre: {}".format(scaled_genre))
# d1.append(data)
# outer_data = np.asarray(d1)

# result = model.predict(outer_data, batch_size=1, verbose=0)
# print(result)
# intified = conv.one_hot_to_int(result[0])
# as_genre = conv.number_to_label(intified)
# descaled_actual_genre = conv.one_hot_to_int(scaled_genre)
# print(descaled_actual_genre)
# d.debug("Predicted genre: {}. Actual: {}".format(as_genre, conv.number_to_label(descaled_actual_genre)))
# print(conv.convert_genre("Indie"))