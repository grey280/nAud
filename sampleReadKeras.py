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



# Testing code
this_song = random.choice(list(tracks.keys())) # picks a random track to look at
this_song_data = tracks.get(this_song)

sample = wav.read(this_song)
d.debug(sample)
d.debug(this_song_data)

genre = conv.convert_genre(this_song_data.get("genre", "Unknown"))
d.debug(genre)

d.debug(conv.genre_to_label(conv.descale_genre(genre)))