from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import plistlib
import gdebug

# Variables
debug_mode = 2 # 0: silent, 1: errors only, 2: normal, 3: verbose
batch_size = 100
input_data = "data/iTunes.plist"

d = gdebug.Debugger(debug_level = debug_mode)

# Helper Functions
def string_to_float(original_name):
	# Convert a string to a float: not a *strict* mapping algorithm, but probably close enough to work
	d.debug("Converting {} to float.".format(original_name))
	multiplier = 0.01
	outvar = 0.1
	for c in original_name:
		outvar = outvar + (multiplier * (ord(c) - 70))
		multiplier = multiplier * 0.1
	d.debug("		Result: {}".format(abs(outvar)))
	return abs(outvar)

def scale_bit_rate(bit_rate):
	# Scale bitrate into a float; using 1500 as the peak value bc 1411 is the highest in my library
	d.debug("Scaling bit rate: {}".format(bit_rate))
	return float(bit_rate/1500)

def scale_year(year):
	# Scale year into a float; using 2016 as it's now, abs as some people have REALLY old music
	d.debug("Scaling year: {}".format(year))
	return abs(float(year/2016))

def scale_rating(rating):
	# Given a rating in /100 format, convert it to [0,1] range
	d.debug("Scaling rating: {}".format(rating))
	return float(rating/100)

# Data handling class
class Data_set:
	start = 0
	label = []
	def __init__(self, inpt):
		# initializer converts from the dictionary to an array, so I can iterate more easily
		self.data=[]
		for track, dt in inpt.items():
			self.data.append(dt)
		d.debug("Initializing data set object")

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

# Input handling
d.debug("Start: read plist")
tracks = plistlib.readPlist(input_data)["Tracks"]
d.debug("End: read plist")

data_set = Data_set(tracks)
d.debug("Data set created.")

# Build Keras model; based on one of the tutorial ones bc why not
model = Sequential()
model.add(Dense(64, input_dim=5, init='uniform')) # 5-dim input: genre,year,bit_rate,artist,title, float-ified
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

d.debug("Model and SGD prepared.")

# model.fit(X-train, y_train, nb_epoch=20, batch_size=16)
# score = model.evaluate(X_test, y_test, batch_size=16)