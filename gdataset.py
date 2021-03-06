# gdataset: Dataset class providing helper functions and sample management for Genre ID.

import numpy 			as np
import scipy.io.wavfile	as wav
import random

import gdebug
import gconvert			as conv

class Dataset:
	# Having a class to handle the dataset makes a lot of things easier.
	# Basically, hand it something opened by plistlib and it'll parse it out nice and pretty-like.
	start = 0
	test_start = 0
	def __init__(self, inpt, do_random=False, sample_duration=15, start_point=0, vstack_split=35, log_level=2, train_size=0.75, train_to="Genre"):
		# Big ol' initializer function - defaults for everything, but lots of settings available.
		# (Maybe one day I'll learn to use **kwargs. Today is not that day.)
		self.input_values = inpt
		self.locations=[]
		self.test_locations=[]
		self.rand = do_random
		self.sample_duration = sample_duration
		self.start_point = start_point
		self.vstack_split_size = vstack_split
		self.d = gdebug.Debugger(debug_level = log_level)
		self.train_to = train_to
		if self.train_to == "Genre":
			self.output_count = conv.number_of_genres
		elif self.train_to == "Bitrate":
			self.output_count = 1024 # arbitrary, okay
		temp_locs = []
		for track, data in inpt.items():
			# self.locations.append(track)
			temp_locs.append(track)
		random.shuffle(temp_locs) # so you don't get a whole blob of one genre in the test data
		for i in range(len(temp_locs)):
			if i > int(train_size*len(temp_locs)):
				self.test_locations.append(temp_locs[i])
			else:
				self.locations.append(temp_locs[i])

	def shuffle(self):
		# Shuffles the training dataset.
		random.shuffle(self.locations)
		self.start = 0

	def shuffle_tests(self):
		# Shuffles the test dataset.
		random.shuffle(self.test_locations)
		self.test_start = 0

	def parse_track(self, track, data):
		# Handles track parsing - given the track and data, does whatever conversions and loading are necessary
		if self.rand:
			return self.random_parse_track(track, data)
		if self.train_to == "Genre":
			genre_orig = data.get("genre", "Unknown")
			genre = int(conv.convert_genre(genre_orig))
			t_to_data = conv.scale_genre(genre)
		elif self.train_to == "Bitrate":
			br_orig = data.get("bitrate", 128)
			try:
				br_orig.replace('vbr','')
			except:
				pass
			shifted_br = int(br_orig)
			if shifted_br > 1024:
				shifted_br = 1024
			if shifted_br < 1:
				shifted_br = 1
			t_to_data = conv.int_to_one_hot(shifted_br, 1024)
		# Process sample
		sample_data = wav.read(track)
		# self.d.verbose("    Samples: {}".format(len(sample_data[1])))
		data = np.ndarray.flatten(sample_data[1])
		del sample_data
		start_point_calc = self.start_point*44100
		end_point_calc = (self.start_point+self.sample_duration)*44100
		return t_to_data, data[start_point_calc:end_point_calc] # force it to be that size, so the NN doesn't complain

	def random_parse_track(self, track, data):
		# Handles track parsing for the '3 5-second chunks from anywhere in the song' test group.
		genre_orig = data.get("genre", "Unknown")
		genre = int(conv.convert_genre(genre_orig))
		scaled_genre = conv.scale_genre(genre)

		# Process sample
		sample_data = wav.read(track)
		self.d.verbose("    Samples: {}".format(len(sample_data[1])))
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
		# Helper function, does what it says on the tin.
		return len(self.locations)

	def get_test_data_point_count(self):
		# Helper function, does what it says on the tin.
		return len(self.test_locations)

	def get_songs(self):
		# Generator to feed songs.
		while True:
			self.start += 1
			# print("\nget_songs with self.start = {} (/{})".format(self.start, len(self.locations)))
			# print("self.start>=(len(self.locations)-2) : {}>={} : {}".format(self.start, len(self.locations)-2, self.start >= (len(self.locations)-2)))
			if self.start+2 >= len(self.locations):
				self.start = 0
			location = self.locations[self.start]
			data_point = self.input_values.get(location)
			genre, output = self.parse_track(location, data_point)
			out = output.reshape(1, len(output))
			g2 = genre.reshape(1, len(genre))
			yield (out, g2)

	def next_batch_infoless(self, data_point_count):
		# Used for training - gets the next batch without the extra information.
		data_array_feed, answer_array_feed, information_feed = self.next_batch(data_point_count)
		return data_array_feed, answer_array_feed

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
			self.d.progress("Loading tracks",i+1,data_point_count)
			try:
				location = self.locations[self.start]
				data_point = self.input_values.get(location)
				genre, output = self.parse_track(location, data_point)
				if(self.start + 2 >= len(self.locations)):
					self.shuffle()
				if(i%self.vstack_split_size == 0):
					data_feed = np.vstack((data_feed, data_feed_holder))
					self.d.verbose(data_feed_holder.shape)
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

	def next_test_batch(self, data_point_count):
		# Same as next_batch, but pulls from the test data rather than the training data.
		location = self.test_locations[self.test_start]
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
			self.d.progress("Loading test tracks",i+1,data_point_count)
			try:
				location = self.test_locations[self.test_start]
				data_point = self.input_values.get(location)
				genre, output = self.parse_track(location, data_point)
				if(self.test_start + 2 >= len(self.test_locations)):
					self.shuffle_tests()
				if(i%self.vstack_split_size == 0):
					data_feed = np.vstack((data_feed, data_feed_holder))
					self.d.verbose(data_feed_holder.shape)
					del data_feed_holder
				self.test_start += 1
				if(i%self.vstack_split_size==0): # fixes an off-by-vstack_split_size error, because np.empty is *weird*
					data_feed_holder = output
				else:
					data_feed_holder = np.vstack((data_feed_holder,output))
				answer_feed.append(genre)
				information_feed.append(location)
			except ValueError:
				self.test_start += 1
				continue
		data_feed = np.vstack((data_feed,data_feed_holder))
		data_array_feed = np.asarray(data_feed)[1:] # fixes an off-by-one error that you get from the way np.empty works
		answer_array_feed = np.asarray(answer_feed)
		return data_array_feed, answer_array_feed, information_feed