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

## IO settings
test_series_name = "INS1"		# name of the test series - files are saved as test_series_name.iteration.json/hdf5
window_size = 1*44100 			# size of windows to feed
file_to_read = "cache/test/LizNelson_Rainfall_MIX.wav" # path to wav file to read

## Operational settings
do_load_model = False
do_load_weights = False
load_from_previous_trial = True
trial_iteration = 1 			# Which iteration of the trial series are you on? Used to load/save. Starts at 0.
which_iteration = 0 			# Which iteration to load

# Tools
d = gdebug.Debugger(debug_level = log_level)

# Helper functions
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

# Multi-iteration crossing
model = load_model(iteration=which_iteration)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
load_weights(iteration=which_iteration)
d.debug("Neural network loaded.")

test_data = data_handler.get_samples_from_file(file_to_read)
results = []
for sample in test_data:
	score = model.predict(sample, batch_size=1)
	results.append(score)

print("Analyzed {} samples, yielding {} results.".format(len(test_data),len(results)))

for result in results:
	pass
	# print(result)

# Evaluate against test data
# test_data, test_answers, test_info_feed = data_set.next_test_batch(evaluation_data_point_count)
# del test_info_feed
# score = model.evaluate(test_data, test_answers, batch_size=batch_size)
# result = "\nTest {} of {} complete. Loss: {}. Accuracy: {}%".format(i+1, tests_in_series, score[0], score[1]*100)
# test_results.append(result)
# d.debug(result)