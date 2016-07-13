from keras.models 		import Sequential, model_from_json
from keras.layers 		import Dense, Dropout, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping, LearningRateScheduler

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav
import sounddevice		as sd

from pythonosc import osc_message_builder
from pythonosc import udp_client

import gdebug

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

## OSC Settings
osc_ip_address = '167.96.80.141'		# IP address to connect to
osc_port = 8020 					# IP port to connect to

## Operational settings
load_from_previous_trial = True
trial_iteration_to_load = 1
trial_to_load = 0

# Tools
d = gdebug.Debugger(debug_level = log_level)
client = udp_client.UDPClient(osc_ip_address, osc_port)

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

model = load_model(iteration=trial_iteration_to_load)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
load_weights(iteration=trial_iteration_to_load)

initial_data_feed = sd.rec(sample_duration*44100, samplerate=44100, channels=1, blocking=True)
sd.wait()
data_feed = np.asarray(initial_data_feed)
data_feed = np.reshape(data_feed, (1, sample_duration*44100))

while True:
	result = model.predict(data_feed, batch_size=1, verbose=0)
	## Send OSC Message
	msg = osc_message_builder.OscMessageBuilder(address = "/tuio2/tok")
	msg.add_arg(0, arg_type="i")
	msg.add_arg(10003, arg_type="i")
	msg.add_arg(result[0][0], arg_type="f") #first value
	msg.add_arg(result[0][1], arg_type="f") #second value
	msg.add_arg(result[0][2], arg_type="f") #third value
	msg = msg.build()
	client.send(msg)
	
	## Get next one to feed
	new_data_feed = sd.rec(1*44100, samplerate=44100, channels=1, blocking=True)
	sd.wait()
	new_data_feed = np.reshape(new_data_feed, (1, 1*44100))
	data_feed = data_feed[:,44100:]
	data_feed = np.hstack((data_feed, new_data_feed))
	
