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

# TensorFlow Helper Funtions
def create_placeholders(batch_size):
	debug("Creating placeholders with batch size {}".format(batch_size))
	input_placeholder = tf.placeholder(tf.float32, shape=(6, batch_size))
	# make it a single placeholder that's wide, reads in each value as part of that one tensor
	# genre_placeholder = tf.placeholder(tf.float32, shape=())
	# year_placeholder = tf.placeholder(tf.float32)
	# bitrate_placeholder = tf.placeholder(tf.float32)
	# artist_placeholder = tf.placeholder(tf.float32)
	# title_placeholder = tf.placeholder(tf.float32)

	rating_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
	# return genre_placeholder, year_placeholder, bitrate_placeholder, artist_placeholder, title_placeholder
	return input_placeholder, rating_placeholder


def fill_feed_dict(data_set, input_pl, rating_pl):
	# TODO insert code to actually fill in the various _feed variables
	#		They should each be a tensor of values
	debug("Filling feed dictionary based on various inputs.")
	feed_dict = {
		input_pl: input_feed,
		rating_pl: rating_feed,
	}
	return feed_dict

class Data_set(inpt):
	start = 0
	label = []
	def __init__(self):
		self.data = inpt


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

