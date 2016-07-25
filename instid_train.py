# instid_train: Trains a neural network for instrument identification.

from keras.models 		import Sequential, model_from_json
from keras.layers 		import Dense, Dropout, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping, LearningRateScheduler

import numpy 			as np
import random

import gdebug
import data_handler

# Settings
## Debug Settings
log_level = 2 					# 0: silent, 1: errors only, 2: normal, 3: verbose

## Neural Network settings
batch_size = 16
epoch_count = 50
data_point_count = 2500 		# number of data points to use for training
evaluation_data_point_count = 750 	# number of data points to evaluate against
shuffle_at_epoch = True 		# shuffle the dataset at each epoch?
NN_validation_split = 0.1 		# fraction of data to be held out as validation data, 0.<x<1
early_stopping_patience = 3 	# how many epochs without improvement it'll go before stopping

## IO settings
test_series_name = "default"	# name of the test series - files are saved as test_series_name.iteration.json/hdf5
tests_in_series = 3 			# number of tests to run in this series
window_size = 4*44100 			# size of windows to feed
samples_in_parallel = 4 		# number of files to be reading in parallel. (1<=n<=4)

## Operational settings
do_load_model = False
do_load_weights = False
load_from_previous_trial = False
trial_iteration = 0 			# Which iteration of the trial series are you on? Used to load/save. Starts at 0.
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
		outpath = "output/{}.{}.{}.json".format(path, trial_iteration, iteration)
		json_string = model.to_json()
		open(outpath, 'w+').write(json_string)
		d.debug('Finished writing model to disk.')

def save_weights(model, iteration, path=test_series_name):
	# Saves the weights - just a quick function to save some time
	if do_save:
		outpath = "output/{}.{}.{}.hdf5".format(path, trial_iteration, iteration)
		model.save_weights(outpath)
		d.debug("Finished writing weights to disk.")

# Multi-iteration crossing
test_results = []

for i in range(tests_in_series):
	d.debug("Test {} of {}".format(i+1, tests_in_series))

	# Build the model, either from scratch or from disk
	if not do_load_model:
		model = Sequential()
		model.add(Dense(512, input_dim=window_size , init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(64, init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(32, init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(16, init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(8, init='uniform'))
		model.add(Activation('tanh'))
		model.add(Dropout(0.5))
		model.add(Dense(4, init='uniform'))
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
		if d.debug_level == 3:
			NN_log_level = 2
		elif d.debug_level == 2:
			NN_log_level = 1
		else:
			NN_log_level = 0

		change_lr = LearningRateScheduler(scheduler)
		early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

		# model.fit(data_feed, answer_feed, nb_epoch=epoch_count, batch_size=batch_size, shuffle=shuffle_at_epoch, validation_split=NN_validation_split, verbose=NN_log_level, callbacks=[early_stopping, change_lr])
		model.fit_generator(data_handler.feed_samples(window_length=window_size, samples_in_parallel=samples_in_parallel), samples_per_epoch=data_point_count, nb_epoch=epoch_count, verbose=NN_log_level, callbacks=[early_stopping, change_lr], validation_data=data_handler.feed_samples(window_length=window_size, samples_in_parallel=samples_in_parallel), nb_val_samples=evaluation_data_point_count)
		d.debug("Fit complete. Preparing to test.")
		d.debug("   Test skipped - code not written. Oh well.")

	# Evaluate against test data
	# test_data, test_answers, test_info_feed = data_set.next_test_batch(evaluation_data_point_count)
	# del test_info_feed
	# score = model.evaluate(test_data, test_answers, batch_size=batch_size)
	# result = "\nTest {} of {} complete. Loss: {}. Accuracy: {}%".format(i+1, tests_in_series, score[0], score[1]*100)
	# test_results.append(result)
	# d.debug(result)

	save_model(model, i)
	save_weights(model, i)

# for result in test_results:
# 	d.debug(result)