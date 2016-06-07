import plistlib
import tensorflow	as tf

# Variables
debug_mode = False
batch_size = 100
input_data = "data/iTunes.plist"

# Helper Functions
def debug(string):
	# I'm lazy, okay?
	if debug_mode:
		print(string)

def string_to_float(original_name):
	# Convert a string to a float: not a *strict* mapping algorithm, but probably close enough to work
	debug("Converting {} to float.".format(original_name))
	multiplier = 0.01
	outvar = 0.1
	for c in original_name:
		outvar = outvar + (multiplier * (ord(c) - 70))
		multiplier = multiplier * 0.1
	debug("		Result: {}".format(abs(outvar)))
	return abs(outvar)

def scale_bit_rate(bit_rate):
	# Scale bitrate into a float; using 1500 as the peak value bc 1411 is the highest in my library
	debug("Scaling bit rate: {}".format(bit_rate))
	return float(bit_rate/1500)

def scale_year(year):
	# Scale year into a float; using 2016 as it's now, abs as some people have REALLY old music
	debug("Scaling year: {}".format(year))
	return abs(float(year/2016))

def scale_rating(rating):
	# Given a rating in /100 format, convert it to [0,1] range
	debug("Scaling rating: {}".format(rating))
	return float(rating/100)

# Data handling class
class Data_set(inpt):
	start = 0
	label = []
	def __init__(self):
		# initializer converts from the dictionary to an array, so I can iterate more easily
		self.data=[]
		for track, dt in inpt:
			self.data.append(dt)

	def next_batch(self, batch_size):
		if (start+batch_size)>len(data):
			start = 0 # reset for next epoch, I suppose?
			return # you're done! this will probably crash at the moment but oh well
		# expected return: input_feed, rating_feed
		input_feed = []
		rating_feed = []
		for i in range(batch_size):
			data_point = self.data[i+start]
			genre = data_point.get("Genre", "Unknown genre")
			year = data_point.get("Year", "unknown")
			bit_rate = data_point.get("Bit Rate", "unknown")
			artist = data_point.get("Artist", "unknown artist")
			title = data_point.get("Name", "Unknown name")
			rating = data_point.get("Rating", 0)
			rating = scale_rating(rating)
			this_one = [genre, year, bit_rate, artist, title]
			input_feed.append(this_one)
			rating_feed.append(rating)
		start += batch_size


# TensorFlow Helper Funtions
def create_placeholders(batch_size):
	debug("Creating placeholders with batch size {}".format(batch_size))
	input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 6))
	# make it a single placeholder that's wide, reads in each value as part of that one tensor
	# used to be lots of placeholders, if I didn't do it like that:
		# genre_placeholder = tf.placeholder(tf.float32, shape=())
		# year_placeholder = tf.placeholder(tf.float32)
		# bitrate_placeholder = tf.placeholder(tf.float32)
		# artist_placeholder = tf.placeholder(tf.float32)
		# title_placeholder = tf.placeholder(tf.float32)

	rating_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
	return input_placeholder, rating_placeholder


def fill_feed_dict(data_obj, input_pl, rating_pl):
	# TODO insert code to actually fill in the various _feed variables
	#		They should each be a tensor of values
	debug("Filling feed dictionary, batch size: {}".format(batch_size))
	input_feed, rating_feed = data_obj.next_batch(batch_size)
	feed_dict = {
		input_pl: input_feed,
		rating_pl: rating_feed,
	}
	return feed_dict


tracks = plistlib.readPlist(input_data)["Tracks"]
debug("Read plist")

# [x] There are 55 genres in my iTunes library, use that as the scale (alphabetical order?)
#		Just gonna use the same function as I'm using for artist/title
# [x] Years can be edited assuming they range 0-2016, scale down to [0,1]
# [x] Bit rate can be scaled from 0kbps to 1500 kbps; one thing has 8 kbps.
#		Might need a boolean switch for variable bitrate
# [x] For artists, write a function that maps the alphabet?
# [x] Do the same for the song title

# Ratings go from 0 to 5, inclusive

input_placeholder, rating_placeholder = create_placeholders(batch_size)
debug("Placeholders generated")

