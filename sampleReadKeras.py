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
debug_mode = 2 # 0: silent, 1: errors only, 2: normal, 3: verbose
batch_size = 100
input_data = "cache/data.plist"

# Tools
d = gdebug.Debugger(debug_level = debug_mode)

# Doing the thing
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
data_feed = [] # [mapped_title, mapped_artist, mapped_year, mapped_bitrate, sample_data (extended out)]
answer_feed = [] # mapped_genre
sample_count_feed = []
for track, data in tracks.items():
	title = data.get("title", "unknown")
	title = conv.string_to_float(title)
	artist = data.get("artist", "unknown")
	artist = conv.string_to_float(artist)
	year = data.get("year", 2016)
	year = conv.scale_year(year)
	bitrate = data.get("bit_rate", 128)
	bitrate = conv.scale_bit_rate(bitrate)
	sample_data = wav.read(track)
	sample_count_feed.append(sample_data[0])
	genre = data.get("genre", "Unknown")
	genre = conv.convert_genre(genre)
	answer_feed.append(genre)
	sample_data = np.ndarray.flatten(sample_data[1])
	output = [title, artist, year, bitrate, sample_data]
	data_feed.append(output)

