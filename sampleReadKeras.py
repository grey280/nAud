#################### TODO #####################
# [x] Speed it up - use numpy
# [x] Saving - implement stuff so I don't need to retrain every time
# [ ] Testing - verify that it's actually accurate
# [ ] Parsing - write stuff so I can throw a song at it and get a genre back, via command line?
###############################################

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav

import gdebug
import gconvert			as conv

# Settings
debug_mode = 2 # 0: silent, 1: errors only, 2: normal, 3: verbose
batch_size = 64*4 # The NN itself will use a batch size of 16, for now; this is the size of the data-parsing batch
NN_batch_size = 16 # Size of batch the NN will use within each sub-epoch
epoch_count = 1
sub_epoch_count = 25 # NN epochs per dataset epoch
input_data = "cache/data.plist"

weights_path = "output/genre_model.json"
model_path = "output/genre_weights.hdf5"
load_model = False
load_weights = False
do_train = True
do_save = True

# Tools
d = gdebug.Debugger(debug_level = debug_mode)

# Helper functions
def parse_track(track, data):
	d.debug("  Parsing track: {}".format(track))
	# Process metadata
	# title_orig = data.get("title", "unknown")
	# title = conv.string_to_int(title_orig)
	# artist_orig = data.get("artist", "unknown")
	# artist = conv.string_to_int(artist_orig)
	# year = int(data.get("year", 2016))
	# bitrate = int(data.get("bit_rate", 128))
	genre_orig = data.get("genre", "Unknown")
	genre = int(conv.convert_genre(genre_orig))
	scaled_genre = conv.scale_genre(genre)

	# Process sample
	sample_data = wav.read(track)
	d.verbose("    Samples: {}".format(len(sample_data[1])))
	# data = [int(val) for sublist in sample_data[1] for val in sublist]
	data = np.ndarray.flatten(sample_data[1])
	del sample_data
	# output = [title, artist, year, bitrate]
	# d.verbose("    Sample list kind: {}".format(type(data)))
	# output.extend(data)

	return scaled_genre, data[:441000] # force it to be that size, so the NN doesn't complain

class Dataset:
	# TODO: implement a way for this to keep some data points aside as test data
	start = 0
	def __init__(self, inpt):
		self.data=[]
		self.locations=[]
		for track, data in inpt.items():
			self.locations.append(track)
			self.data.append(data)
		d.debug("Initializing data set object")
	def next_batch(self, batch_size):
		if (self.start+batch_size)>len(self.data):
			self.start = 0 # reset for next epoch, I suppose?
			#return # you're done! this will probably crash at the moment but oh well. update: yeah it crashed
		# expected return: data_feed, answer_feed
		data_feed = []
		answer_feed = []
		for i in range(batch_size):
			data_point = self.data[i+self.start]
			location = self.locations[i+self.start]
			genre, output = parse_track(location, data_point)
			
			data_feed.append(output)
			answer_feed.append(genre)
		self.start += batch_size
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
# And have it training towards the genres. Which I should probably convert into a numerical system, set in stone,
#	so that it can be consistent across trainings and whatnot. I'll go write that.

data_set = Dataset(tracks)
d.debug("Dataset built.")

if not load_model:
	model = Sequential()
	model.add(Dense(64, input_dim=441000 , init='uniform')) # number of data points being fed in: 4 metatags, 441000 samples (10 sec@44.1kHz)
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
	model = open(model_path, 'r').read()
	model = model_from_json(model)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	d.debug("Model loaded and SGD prepared.")
	if load_weights:
		model.load_weights(weights_path)
		d.debug("Weights loaded.")
if do_train:
	for i in range(epoch_count):
		data_feed, answer_feed = data_set.next_batch(batch_size)
		d.debug("Meta-epoch {} of {}.".format(i, epoch_count))
		model.fit(data_feed, answer_feed, nb_epoch=sub_epoch_count, batch_size=NN_batch_size)

d.debug("Fit complete. Preparing to test.")
test_data, test_answers = data_set.next_batch(batch_size*4)
score = model.evaluate(test_data, test_answers, batch_size=16)
d.debug("")
d.debug("Test complete. Loss: {}. Accuracy: {}%".format(score[0], score[1]*100))

if do_save:
	json_string = model.to_json()
	model.save_weights(weights_path)
	open(model_path, 'w+').write(json_string)
	d.debug('Finished writing weights and model to disk.')


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