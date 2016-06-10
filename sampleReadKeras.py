from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav

import gdebug
import gconvert			as conv

import random			# for testing only; once the this_song = random.choice() line is gone, take this out
from sys import getsizeof # used for profiling during testing only

# Settings
debug_mode = 3 # 0: silent, 1: errors only, 2: normal, 3: verbose
batch_size = 100
input_data = "cache/data.plist"

# Tools
d = gdebug.Debugger(debug_level = debug_mode)

# Helper functions
def parse_track(track, data):
	global debug_counter
	d.verbose("  Parsing track: {}".format(track))
	debug_counter += 1
	d.verbose("    Track {} of {}".format(debug_counter, total_count))

	# Process metadata
	title_orig = data.get("title", "unknown")
	title = conv.string_to_int(title_orig)
	artist_orig = data.get("artist", "unknown")
	artist = conv.string_to_int(artist_orig)
	year = int(data.get("year", 2016))
	bitrate = int(data.get("bit_rate", 128))
	genre_orig = data.get("genre", "Unknown")
	genre = int(conv.convert_genre(genre_orig))
	# answer_feed.append(genre)

	# Process sample
	sample_data = wav.read(track)
	# if debug_counter == 1:
	# 	d.debug(sample_data[1][:500])
	d.verbose("    Samples: {}".format(len(sample_data[1])))
	# sample_rate_feed.append(sample_data[0])
	# sample_data = np.ndarray.flatten(sample_data[1]) # numpy is apparently the memory hog, so try without it
	data = [int(val) for sublist in sample_data[1] for val in sublist]
	del sample_data
	output = [title, artist, year, bitrate]
	d.debug("    Sample list kind: {}".format(type(data)))
	output.extend(data)
	return genre, output

# @profile
def parse_tracks(tracks):
	d.debug("Building feeds.")
	data_feed = [] # [mapped_title, mapped_artist, mapped_year, mapped_bitrate, sample_data (extended out)]
	answer_feed = [] # mapped_genre
	global debug_counter
	for track, data in tracks.items():
		genre, output = parse_track(track, data)
		answer_feed.append(genre)
		data_feed.append(output)
		del output
		del genre
		d.debug("    Current size of data_feed: {}".format(getsizeof(data_feed)))
		# if debug_counter >= 10:
		# 	break
	return data_feed, answer_feed


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

# Feed builders


# sample_rate_feed = []

debug_counter = 0
total_count = len(tracks)
data_feed, answer_feed = parse_tracks(tracks)


# np.save("tempdump1.npy", answer_feed)
# d.debug("Saved answer feed to disk.")

# np.save("tempdump.npy", data_feed)
# d.debug("Saved data feed to disk.")

d.debug("Feeds constructed.")
d.debug("{} {} {} {} {} {}".format(data_feed[0][0], data_feed[0][1], data_feed[0][2], data_feed[0][3], data_feed[0][4], data_feed[0][5]))
d.debug("Length of first data_feed item: {}".format(len(data_feed[0])))
# model = Sequential()
# model.add(Dense(64, input_dim=7 , init='uniform')) # 5-dim input: genre,year,bit_rate,artist,title, float-ified
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('tanh'))
# model.add(Dropout(0.5))
# model.add(Dense(6, init='uniform')) # 6 because the ratings are range(0,5)
# model.add(Activation('softmax'))

# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# d.debug("Model and SGD prepared.")

# model.fit(X, y, nb_epoch=50, batch_size=32)
# d.debug("Fit complete. Preparing to test.")
# model.summary()
# score = model.evaluate(X_test, y_test, batch_size=16)
# d.debug("")
# d.debug("Test complete. Loss: {}. Accuracy: {}%".format(score[0], score[1]*100))