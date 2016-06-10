from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav

import gdebug
import gconvert			as conv

import random			# for testing only; once the this_song = random.choice() line is gone, take this out

# Settings
debug_mode = 3 # 0: silent, 1: errors only, 2: normal, 3: verbose
batch_size = 100
input_data = "cache/data.plist"

# Tools
d = gdebug.Debugger(debug_level = debug_mode)

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
d.debug("Building feeds.")
data_feed = [] # [mapped_title, mapped_artist, mapped_year, mapped_bitrate, sample_data (extended out)]
answer_feed = [] # mapped_genre
# sample_rate_feed = []

debug_counter = 0
total_count = len(tracks)

for track, data in tracks.items():
	d.verbose("  Parsing track: {}".format(track))
	debug_counter += 1
	d.verbose("  Track {} of {}".format(debug_counter, total_count))
	title = data.get("title", "unknown")
	title = conv.string_to_float(title)
	artist = data.get("artist", "unknown")
	artist = conv.string_to_float(artist)
	year = data.get("year", 2016)
	year = conv.scale_year(year)
	bitrate = data.get("bit_rate", 128)
	bitrate = conv.scale_bit_rate(bitrate)
	sample_data = wav.read(track)
	# if debug_counter == 1:
	# 	d.debug(sample_data[1][:500])
	d.verbose("    Samples: {}".format(len(sample_data[1])))
	# sample_rate_feed.append(sample_data[0])
	genre = data.get("genre", "Unknown")
	genre = conv.convert_genre(genre)
	answer_feed.append(genre)
	# sample_data = np.ndarray.flatten(sample_data[1]) # numpy is apparently the memory hog, so try without it
	sample_data = [val for sublist in sample_data[1] for val in sublist]
	output = [title, artist, year, bitrate]
	#sample_data = [float(x)/32768 for x in sample_data] # scale into float range
	# okay rescaling that is HELLA SLOW, let's do the inverse
	output = [int(x*32768) for x in output]
	output.extend(sample_data)
	data_feed.append(output)

	del output
	del sample_data
	if debug_counter >= 10:
		break

np.save("tempdump1.npy", answer_feed)
d.debug("Saved answer feed to disk.")

np.save("tempdump.npy", data_feed)
d.debug("Saved data feed to disk.")

d.debug("Feeds constructed.")
d.debug("{} {} {} {} {} {}".format(data_feed[0][0], data_feed[0][1], data_feed[0][2], data_feed[0][3], data_feed[0][4], data_feed[0][5]))
print(type(data_feed[0][5]))
d.debug(data_feed[0][:5000])
# d.verbose("Count feed:{}".format(sample_rate_feed))
d.debug(len(data_feed[0]))
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