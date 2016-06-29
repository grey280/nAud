from keras.models 		import Sequential, model_from_json
from keras.layers 		import Dense, Dropout, Activation
from keras.optimizers 	import SGD
from keras.callbacks 	import EarlyStopping, LearningRateScheduler

import plistlib
import numpy 			as np
import scipy.io.wavfile	as wav
import random
import time

import gdebug
import gconvert			as conv

# Settings
## Debug Settings
log_level = 2 							# 0: silent, 1: errors only, 2: normal, 3: verbose





class Dataset:
	# Having a class to handle the dataset makes a lot of things easier.
	# Basically, hand it something opened by plistlib and it'll parse it out nice and pretty-like.
	start = 0
	def __init__(self, inpt, do_random=False, sample_duration=15, start_point=0, vstack_split=35):
		self.input_values = inpt
		self.locations=[]
		self.rand = do_random
		self.sample_duration = sample_duration
		self.start_point = start_point
		self.vstack_split_size = vstack_split
		for track, data in inpt.items():
			self.locations.append(track)
		# d.debug("Initializing data set object")
	def shuffle(self):
		# Shuffles the dataset. May be expanded in the future to better handle 'holding out test data' functionality.
		random.shuffle(self.locations)
		self.start = 0

	def parse_track(self, track, data):
		# Handles track parsing - given the track and data, does whatever conversions and loading are necessary
		if self.rand:
			return self.random_parse_track(track, data)
		genre_orig = data.get("genre", "Unknown")
		genre = int(conv.convert_genre(genre_orig))
		scaled_genre = conv.scale_genre(genre)

		# Process sample
		sample_data = wav.read(track)
		# d.verbose("    Samples: {}".format(len(sample_data[1])))
		data = np.ndarray.flatten(sample_data[1])
		del sample_data
		start_point_calc = start_point*44100
		end_point_calc = (start_point+sample_duration)*44100
		return scaled_genre, data[start_point_calc:end_point_calc] # force it to be that size, so the NN doesn't complain

	def random_parse_track(self, track, data):
		# Handles track parsing for the '3 5-second chunks from anywhere in the song' test group.
		genre_orig = data.get("genre", "Unknown")
		genre = int(conv.convert_genre(genre_orig))
		scaled_genre = conv.scale_genre(genre)

		# Process sample
		sample_data = wav.read(track)
		d.verbose("    Samples: {}".format(len(sample_data[1])))
		total_samples = len(sample_data[1])
		data = np.ndarray.flatten(sample_data[1])
		del sample_data
		duration = ((self.sample_duration/3)*44100)
		if duration >= total_samples:
			raise ValueError('Song is not long enough.')
		start_point_1 = int(random.randrange(total_samples - duration))
		start_point_2 = int(random.randrange(total_samples - duration))
		start_point_3 = int(random.randrange(total_samples - duration))
		data_1 = data[start_point_1:int(start_point_1 + duration)]
		data_2 = data[start_point_2:int(start_point_2 + duration)]
		data_3 = data[start_point_3:int(start_point_3 + duration)]
		return scaled_genre, np.concatenate((data_1, data_2, data_3))

	def get_data_point_count(self):
		# This function may get expanded in the future to better handle 'holding out test data' functionality.
		return len(self.locations)

	def next_batch(self, data_point_count):
		# Loads the next batch - with optimizations, this can actually handle batch sizes in the (0,2000) range
		# pretty well - don't actually know how big it gets without trouble, that's all I've tested.
		# Of course, with big batches, loading *does* get slow, but there's not much you can do about that.
		location = self.locations[self.start]
		data_point = self.input_values.get(location)
		genre, output = self.parse_track(location, data_point)
		answer_feed = [genre]
		information_feed = [location]
		try:
			output = output.asarray()
		except:
			pass
		data_feed_holder = output
		data_feed = np.empty((44100*self.sample_duration,),dtype='int16')
		for i in range(1, data_point_count):
			d.progress("Loading tracks",i+1,data_point_count)
			try:
				location = self.locations[self.start]
				data_point = self.input_values.get(location)
				genre, output = self.parse_track(location, data_point)
				if(self.start + 2 >= len(self.locations)):
					self.shuffle()
				if(i%self.vstack_split_size == 0):
					data_feed = np.vstack((data_feed, data_feed_holder))
					d.verbose(data_feed_holder.shape)
					del data_feed_holder
				self.start += 1
				if(i%self.vstack_split_size==0): # fixes an off-by-vstack_split_size error, because np.empty is *weird*
					data_feed_holder = output
				else:
					data_feed_holder = np.vstack((data_feed_holder,output))
				answer_feed.append(genre)
				information_feed.append(location)
			except ValueError:
				self.start += 1
				continue
		data_feed = np.vstack((data_feed,data_feed_holder))
		data_array_feed = np.asarray(data_feed)[1:] # fixes an off-by-one error that you get from the way np.empty works
		answer_array_feed = np.asarray(answer_feed)
		return data_array_feed, answer_array_feed, information_feed