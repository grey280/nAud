# Things to try
# * Various transforms of the input data to a different setup
# * Pull samples from a different part of the song - :30-:40 instead of :00-:10, or something
# * Restructure the neural network model

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav
import random

import gdebug
import gconvert			as conv

# Settings
## Debug Settings
debug_mode = 2 # 0: silent, 1: errors only, 2: normal, 3: verbose

## Neural Network settings
batch_size = 16
epoch_count = 50
data_point_count = 0 # number of data points to use for training; set to 0 for 'all'
evaluation_data_point_count = 256 # number of data points to evaluate against; set to 0 for 'all'

## IO settings
input_data = "cache/data.plist"
weights_file_name = "genre_model2.json"
model_file_name = "genre_weights2.hdf5"

## Operational settings
load_model = False
load_weights = False
do_train = True
do_save = True

# Tools
d = gdebug.Debugger(debug_level = debug_mode)

# Helper functions
def parse_track(track, data):
	
	genre_orig = data.get("genre", "Unknown")
	genre = int(conv.convert_genre(genre_orig))
	scaled_genre = conv.scale_genre(genre)

	# Process sample
	sample_data = wav.read(track)
	d.verbose("    Samples: {}".format(len(sample_data[1])))
	data = np.ndarray.flatten(sample_data[1])
	del sample_data

	return scaled_genre, data[:441000] # force it to be that size, so the NN doesn't complain

def save_model(model, path=model_file_name):
	if do_save:
		path = "output/{}".format(path)
		json_string = model.to_json()
		open(path, 'w+').write(json_string)
		d.debug('Finished writing model to disk.')

def save_weights(model, path=weights_file_name):
	if do_save:
		path = "output/{}".format(path)
		model.save_weights(path)
		d.debug("Finished writing weights to disk.")

class Dataset:
	# TODO: implement a way for this to keep some data points aside as test data
	start = 0
	def __init__(self, inpt):
		self.input_values = inpt
		self.locations=[]
		for track, data in inpt.items():
			self.locations.append(track)
		d.debug("Initializing data set object")
	def shuffle(self):
		random.shuffle(self.locations)
		self.start = 0

	def get_data_point_count(self):
		return len(self.locations)

	def next_batch(self, data_point_count):
		if(self.start+data_point_count+2 >= len(self.locations)):
			self.shuffle()
		# expected return: data_feed, answer_feed
		location = self.locations[self.start]
		data_point = self.input_values.get(location)
		genre, output = parse_track(location, data_point)
		answer_feed = []
		answer_feed.append(genre)
		try:
			temp_output = output.asarray()
		except:
			temp_output = output
		data_feed = temp_output
		d.verbose("Data point size: {}".format(data_feed.shape))
		for i in range(1, data_point_count):
			location = self.locations[i+self.start]
			data_point = self.input_values.get(location)
			genre, output = parse_track(location, data_point)
			d.progress("  Parsed track: {}".format(location),i,data_point_count)
			d.verbose("Data point size: {}".format(output.shape))
			data_feed = np.vstack((data_feed,output)) # TODO: fix this
								# it works, but it gets slower and slower over time until it's
								# just downright ungodly. probably because np.vstack doesnt't
								# edit in place, it does a full copy and edits the copy.
								# so increasing efficiency options:
								# * find something that edits in place, I don't need copying
								# * copy by batches - combine into fives or something?
			answer_feed.append(genre)
			if (self.start+i+2)>len(self.locations):
				self.start = 0 # start over at the beginning
		self.start += data_point_count
		answer_array_feed = np.asarray(answer_feed)
		data_array_feed = np.asarray(data_feed)
		return data_array_feed, answer_array_feed


# Import data
d.debug("Start: read plist")
tracks = plistlib.readPlist(input_data)
d.debug("End: read plist")
# Example key/value pair, for ease of reading: cache/2013.OK Go.Twelve Months of OK Go.Get Over It (Elevator Version).wav: {'title': 'Get Over It (Elevator Version)', 'year': 2013, 'play_count': 0, 'bit_rate': 320, 'genre': 'Indie Rock', 'artist': 'OK Go', 'skip_count': 0, 'rating': 40}
# So, for the neural network, we'll feed it:
#	Title
#	Artist
#	Year
#	Bitrate?
#	Sample data
# And it's trying to identify the genre.

data_set = Dataset(tracks)
d.debug("Dataset built.")
d.verbose("Dataset size: {}".format(data_set.get_data_point_count()))
if data_point_count == 0:
	data_point_count = data_set.get_data_point_count()
if evaluation_data_point_count == 0:
	evaluation_data_point_count = data_set.get_data_point_count()

# Build the model, either from scratch or from disk
if not load_model:
	model = Sequential()
	model.add(Dense(128, input_dim=441000 , init='uniform')) # number of data points being fed in: 4 metatags, 441000 samples (10 sec@44.1kHz)
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(conv.number_of_genres, init='uniform')) # hopefully this works; keeps it dynamic
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	d.debug("Model and SGD prepared.")
	if debug_mode == 3: # only need to print the model in Verbose mode
		model.summary()
else:
	model = open(model_file_name, 'r').read()
	model = model_from_json(model)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	d.debug("Model loaded and SGD prepared.")
	if load_weights:
		model.load_weights(weights_file_name)
		d.debug("Weights loaded.")
# Training
if do_train:
	data_feed, answer_feed = data_set.next_batch(data_point_count)
	model.fit(data_feed, answer_feed, nb_epoch=epoch_count, batch_size=batch_size)
	d.debug("Fit complete. Preparing to test.")

# Evaluate against test data
test_data, test_answers = data_set.next_batch(evaluation_data_point_count)
score = model.evaluate(test_data, test_answers, batch_size=16)
d.debug("\nTest complete. Loss: {}. Accuracy: {}%".format(score[0], score[1]*100))

save_model(model)
save_weights(model)


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

# new plan: instead of putting the full-loaded dataset into memory as a MASSIVE array,
# just write a function that'll spit out a more manageable chunk at a time, and use that to 
# manually do epochs - wrap the model.fit() function in a loop, giving different training
# data each time the loop runs, and have it only do one epoch at a time. Manual override for epochs,
# allowing for intelligent feed-in of data in a way that doesn't require a bloody TB of memory