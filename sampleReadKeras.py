from keras.models 		import Sequential, model_from_json
from keras.layers 		import Dense, Dropout, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping, LearningRateScheduler

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav
import random

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
# weights_file_name = "nDS.json"			# name of model file to load
# model_file_name = "nDS.hdf5"			# name of weights file to load
test_series_name = "2S1M"				# name of the test series - files are saved as test_series_name.iteration.json/hdf5
tests_in_series = 3 					# number of tests to run in this series
vstack_split_size = 35					# controls the speed/memory usage of loading tracks. 25-50 works well.
start_point = 60 						# seconds into the sample to read ((start_point+sample_duration)<sample length)
sample_duration = 20					# seconds of sample to read ((start_point+sample_duration)<sample length)
do_random_parse = False					# true will use three 5-second clips from random places in the song, rather than a single 15-second block

## Operational settings
do_load_model = False
do_load_weights = False
load_from_previous_trial = False
trial_iteration = 0 					# Which iteration of the trial series are you on? Used to load/save. Starts at 0.
do_train = True
do_save = True

# Tools
d = gdebug.Debugger(debug_level = log_level)

# Helper functions
def scheduler(epoch):
	# Alters the learning rate depending on which epoch you're in.
	if epoch >= 10:
		return 0.01
	elif epoch >= 5:
		return 0.05
	else:
		return 0.1

def parse_track(track, data):
	# Handles track parsing - given the track and data, does whatever conversions and loading are necessary
	if do_random_parse:
		return random_parse_track(track, data)
	
	genre_orig = data.get("genre", "Unknown")
	genre = int(conv.convert_genre(genre_orig))
	scaled_genre = conv.scale_genre(genre)

	# Process sample
	sample_data = wav.read(track)
	total_samples = len(sample_data[1])
	d.verbose("    Samples: {}".format(total_samples))
	start_point_calc = start_point*44100
	end_point_calc = (start_point+sample_duration)*44100
	if end_point_calc >= total_samples:
		raise ValueError('Song is not long enough.')
	data = np.ndarray.flatten(sample_data[1])
	del sample_data
	
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
		load_path = "output/{}.{}.{}.json".format(path, trial_iteration-1, iteration)
	else:
		load_path = "output/{}".format(path)
	model = open(load_path, 'r').read()
	return model_from_json(model)

def load_weights(iteration=0, path=test_series_name):
	global model
	if load_from_previous_trial:
		load_path = "output/{}.{}.{}.hdf5".format(path, trial_iteration-1, iteration)
	else:
		load_path = "output/{}.hdf5".format(path)
	model.load_weights(load_path)

def save_model(model, iteration, path=test_series_name):
	# Saves the model - just a quick function to save some time
	if do_save:
		outpath = "output/{}.{}.json".format(path, iteration)
		if load_from_previous_trial:
			outpath = "output/{}.{}.{}.json".format(path, trial_iteration, iteration)
		json_string = model.to_json()
		open(outpath, 'w+').write(json_string)
		d.debug('Finished writing model to disk.')

def save_weights(model, iteration, path=test_series_name):
	# Saves the weights - just a quick function to save some time
	if do_save:
		outpath = "output/{}.{}.hdf5".format(path, iteration)
		if load_from_previous_trial:
			outpath = "output/{}.{}.{}.hdf5".format(path, trial_iteration, iteration)
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
			d.progress("Loading tracks".format(location),i+1,data_point_count)
			if(i%vstack_split_size==0): # fixes an off-by-vstack_split_size error, because np.empty is *weird*
				data_feed_holder = output
			else:
				data_feed_holder = np.vstack((data_feed_holder,output))
			answer_feed.append(genre)
		data_feed = np.vstack((data_feed,data_feed_holder))
		data_array_feed = np.asarray(data_feed)[1:] # fixes an off-by-one error that you get from the way np.empty works
		answer_array_feed = np.asarray(answer_feed)
		return data_array_feed, answer_array_feed


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

# Multi-iteration crossing
test_results = []

for i in range(tests_in_series):
	d.debug("Test {} of {}".format(i, tests_in_series))

	# Build the model, either from scratch or from disk
	if not do_load_model:
		model = Sequential()
		model.add(Dense(128, input_dim=44100*sample_duration , init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(64, init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(conv.number_of_genres, init='uniform'))
		model.add(Activation('softmax'))

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

		d.debug("Model and SGD prepared.")
		if log_level == 3: # only need to print the model in Verbose mode
			model.summary()
	else:
		model = load_model(iteration=i)
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
		d.debug("Model loaded and SGD prepared.")
		if do_load_weights:
			load_weights(iteration=i)
			d.debug("Weights loaded.")
	# Training
	if do_train:
		data_feed, answer_feed = data_set.next_batch(data_point_count)
		if d.debug_level == 3:
			NN_log_level = 2
		elif d.debug_level == 2:
			NN_log_level = 1
		else:
			NN_log_level = 0

		change_lr = LearningRateScheduler(scheduler)
		early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)
		model.fit(data_feed, answer_feed, nb_epoch=epoch_count, batch_size=batch_size, shuffle=shuffle_at_epoch, validation_split=NN_validation_split, verbose=NN_log_level, callbacks=[early_stopping, change_lr])
		d.debug("Fit complete. Preparing to test.")

	# Evaluate against test data
	test_data, test_answers = data_set.next_batch(evaluation_data_point_count)
	score = model.evaluate(test_data, test_answers, batch_size=batch_size)
	result = "\nTest {} of {} complete. Loss: {}. Accuracy: {}%".format(i, tests_in_series, score[0], score[1]*100)
	test_results.append(result)
	d.debug(result)

	save_model(model, i)
	save_weights(model, i)

for result in test_results:
	d.debug(result)