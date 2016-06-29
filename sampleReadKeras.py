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
import gdataset 		as ds

# Settings
## Debug Settings
log_level = 2 							# 0: silent, 1: errors only, 2: normal, 3: verbose

## Neural Network settings
batch_size = 16
epoch_count = 50
data_point_count = 100 					# number of data points to use for training; set to 0 for 'all'
evaluation_data_point_count = 256 		# number of data points to evaluate against; set to 0 for 'all'
shuffle_at_epoch = True
NN_validation_split = 0.1 				# fraction of data to be held out as validation data, 0.<x<1
early_stopping_patience = 3 			# how many epochs without improvement it'll go before stopping

## IO settings
input_data = "cache/data.plist"
test_series_name = "quickTest"				# name of the test series - files are saved as test_series_name.iteration.json/hdf5
tests_in_series = 3 					# number of tests to run in this series

## Data set settings
vstack_split_size = 25					# controls the speed/memory usage of loading tracks. 25-50 works well.
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
		outpath = "output/{}.0.{}.json".format(path, iteration)
		if load_from_previous_trial:
			outpath = "output/{}.{}.{}.json".format(path, trial_iteration, iteration)
		json_string = model.to_json()
		open(outpath, 'w+').write(json_string)
		d.debug('Finished writing model to disk.')

def save_weights(model, iteration, path=test_series_name):
	# Saves the weights - just a quick function to save some time
	if do_save:
		outpath = "output/{}.0.{}.hdf5".format(path, iteration)
		if load_from_previous_trial:
			outpath = "output/{}.{}.{}.hdf5".format(path, trial_iteration, iteration)
		model.save_weights(outpath)
		d.debug("Finished writing weights to disk.")

# Import data
d.debug("Start: read plist")
tracks = plistlib.readPlist(input_data)
d.debug("End: read plist")
data_set = ds.Dataset(tracks, do_random=do_random_parse, sample_duration=sample_duration, start_point=start_point, vstack_split=vstack_split_size)
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
		model.add(Dense(256, input_dim=44100*sample_duration , init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(128, init='uniform'))
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
		data_feed, answer_feed, information_feed = data_set.next_batch(data_point_count)
		del information_feed
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
	test_data, test_answers, test_info_feed = data_set.next_test_batch(evaluation_data_point_count)
	del test_info_feed
	score = model.evaluate(test_data, test_answers, batch_size=batch_size)
	result = "\nTest {} of {} complete. Loss: {}. Accuracy: {}%".format(i, tests_in_series, score[0], score[1]*100)
	test_results.append(result)
	d.debug(result)

	save_model(model, i)
	save_weights(model, i)

for result in test_results:
	d.debug(result)