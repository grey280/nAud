from keras.models 		import Sequential, model_from_json
from keras.layers 		import Dense, Dropout, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping, LearningRateScheduler

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav
import sounddevice		as sd

import gdebug
import gconvert			as conv
import gdataset  		as ds

# Settings
## Debug Settings
log_level = 2 							# 0: silent, 1: errors only, 2: normal, 3: verbose

## Neural Network settings
batch_size = 16

## IO settings
input_data = "cache/data.plist"
weights_file_name = ".json"		# name of model file to load
model_file_name = ".hdf5"		# name of weights file to load
test_series_name = "BGNN"			# name of the test series - files are saved as test_series_name.iteration.[json/hdf5]
vstack_split_size = 35					# controls the speed/memory usage of loading tracks. 25-50 works well.
start_point = 60 						# seconds into the sample to read ((start_point+sample_duration)<sample length)
sample_duration = 20					# seconds of sample to read ((start_point+sample_duration)<sample length)
do_random_parse = False					# true will use three 5-second clips from random places in the song, rather than a single 15-second block

## Operational settings
load_from_previous_trial = True
trial_iteration_to_load = 0
trial_to_load = 0

# Tools
d = gdebug.Debugger(debug_level = log_level)

# Helper functions
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

# Import data
tracks = plistlib.readPlist(input_data)
data_set = ds.Dataset(tracks, do_random=do_random_parse, sample_duration=sample_duration, start_point=start_point, vstack_split=vstack_split_size, log_level=0)

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
	d.debug("{}/{}/{}/{}/{}/{}/{}/{}/{}".format(information_feed[i],result[0][0],result[0][1],result[0][2],as_int,answer_array_feed[0][0],answer_array_feed[0][1],answer_array_feed[0][2],orig_as_int))

	# Output: song_identifier, likelihood_art, likelihood_pop, likelihood_tradition, was_art, was_pop, was_tradition