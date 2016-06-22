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
evaluation_data_point_count = 256 		# number of data points to evaluate against; set to 0 for 'all'
shuffle_at_epoch = True
NN_validation_split = 0.1 				# fraction of data to be held out as validation data, 0.<x<1
early_stopping_patience = 3 			# how many epochs without improvement it'll go before stopping

## IO settings
input_data = "cache/data.plist"
weights_file_name = "default.json"		# name of model file to load
model_file_name = "default.hdf5"		# name of weights file to load
test_series_name = "default"			# name of the test series - files are saved as test_series_name.iteration.json/hdf5
tests_in_series = 3 					# number of tests to run in this series
vstack_split_size = 35					# controls the speed/memory usage of loading tracks. 25-50 works well.
start_point = 60 						# seconds into the sample to read ((start_point+sample_duration)<sample length)
sample_duration = 15					# seconds of sample to read ((start_point+sample_duration)<sample length)
do_random_parse = True					# true will use three 5-second clips from random places in the song, rather than a single 15-second block

## Operational settings
load_from_previous_trial = False

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
	d.verbose("    Samples: {}".format(len(sample_data[1])))
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
	start_point_1 = int(random.randrange(total_samples - ((sample_duration/3)*44100)))
	start_point_2 = int(random.randrange(total_samples - ((sample_duration/3)*44100)))
	start_point_3 = int(random.randrange(total_samples - ((sample_duration/3)*44100)))
	data_1 = data[start_point_1:int(start_point_1 + ((sample_duration/3)*44100))]
	data_2 = data[start_point_2:int(start_point_2 + ((sample_duration/3)*44100))]
	data_3 = data[start_point_3:int(start_point_3 + ((sample_duration/3)*44100))]
	return scaled_genre, np.concatenate((data_1, data_2, data_3))

def load_model(iteration=0, path=test_series_name):
	if load_from_previous_trial:
		load_path = "output/{}.{}.json".format(path, iteration)
	else:
		load_path = "output/{}".format(model_file_name)
	model = open(load_path, 'r').read()
	return model_from_json(model)

def load_weights(iteration=0, path=test_series_name):
	global model
	if load_from_previous_trial:
		load_path = "output/{}.{}.hdf5".format(path, iteration)
	else:
		load_path = "output/{}.hdf5".format(path)
	model.load_weights(load_path)

def save_model(model, iteration, path=test_series_name):
	# Saves the model - just a quick function to save some time
	if do_save:
		outpath = "output/{}.{}.json".format(path, iteration)
		if load_from_previous_trial:
			outpath = "output/{}.{}.{}.json".format(path, iteration, time.time())
		json_string = model.to_json()
		open(outpath, 'w+').write(json_string)
		d.debug('Finished writing model to disk.')

def save_weights(model, iteration, path=test_series_name):
	# Saves the weights - just a quick function to save some time
	if do_save:
		outpath = "output/{}.{}.hdf5".format(path, iteration)
		if load_from_previous_trial:
			outpath = "output/{}.{}.{}.hdf5".format(path, iteration, time.time())
		model.save_weights(outpath)
		d.debug("Finished writing weights to disk.")

class Dataset:
	# Having a class to handle the dataset makes a lot of things easier.
	# Basically, hand it something opened by plistlib and it'll parse it out nice and pretty-like.
	start = 0
	def __init__(self, inpt):
		self.input_values = inpt
		self.locations=[]
		for track, data in inpt.items():
			self.locations.append(track)
		d.debug("Initializing data set object")
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
			information_feed.append(location)
			self.start += 1
			data_point = self.input_values.get(location)
			genre, output = parse_track(location, data_point)
			d.progress("Loading tracks".format(location),i+1,data_point_count)
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
d.debug("Start: read plist")
tracks = plistlib.readPlist(input_data)
d.debug("End: read plist")
data_set = Dataset(tracks)
d.debug("Dataset built.")
d.verbose("Dataset size: {}".format(data_set.get_data_point_count()))

# Load configuration, if necessary
if data_point_count == 0:
	data_point_count = data_set.get_data_point_count()
if evaluation_data_point_count == 0:
	evaluation_data_point_count = data_set.get_data_point_count()

model = load_model(iteration=i)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
d.debug("Model loaded and SGD prepared.")
load_weights(iteration=i)
d.debug("Weights loaded.")
data_array_feed, answer_array_feed, information_feed = data_set.next_batch(data_point_count)
for i in range(len(data_array_feed)):
	result = model.predict(data_array_feed[i], data_point_count=1, verbose=0)
	d.debug("{}: {}\n  Correct: {}".format(information_feed[i], conv.genre_to_label(conv.one_hot_to_int(result[0])), conv.descale_genre(answer_array_feed[i])))


# specific_song_to_test = "cache/2016.Ten Fé.NOON  189.Elodie.wav"
# that_data = {"genre": "Indie"}

# d1 = []
# scaled_genre, data = parse_track(specific_song_to_test, that_data)
# print("scaled_genre: {}".format(scaled_genre))
# d1.append(data)
# outer_data = np.asarray(d1)

# result = model.predict(outer_data, data_point_count=1, verbose=0)
# print(result)
# intified = conv.one_hot_to_int(result[0])
# as_genre = conv.genre_to_label(intified)
# descaled_actual_genre = conv.one_hot_to_int(scaled_genre)
# print(descaled_actual_genre)
# d.debug("Predicted genre: {}. Actual: {}".format(as_genre, conv.genre_to_label(descaled_actual_genre)))
# print(conv.convert_genre("Indie"))