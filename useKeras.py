from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import plistlib
import gdebug
import numpy 			as np

# Variables
debug_mode = 2 # 0: silent, 1: errors only, 2: normal, 3: verbose
batch_size = 100
input_data = "data/iTunes.plist"

load_model = False
load_weights = False
# The paths are used as where to load from if loading is enabled, and where to save to.
model_path = "output/model.json"
weights_path = "output/weights.hdf5"

d = gdebug.Debugger(debug_level = debug_mode)

# Helper Functions
def string_to_float(original_name):
	# Convert a string to a float: not a *strict* mapping algorithm, but probably close enough to work
	d.verbose("Converting {} to float.".format(original_name))
	multiplier = 0.01
	outvar = 0.1
	for c in original_name:
		outvar = outvar + (multiplier * (ord(c) - 70))
		multiplier = multiplier * 0.1
	d.verbose("		Result: {}".format(abs(outvar)))
	return abs(outvar)

def scale_bit_rate(bit_rate):
	# Scale bitrate into a float; using 1500 as the peak value bc 1411 is the highest in my library
	d.verbose("Scaling bit rate: {}".format(bit_rate))
	return float(bit_rate/1500)

def scale_year(year):
	# Scale year into a float; using 2016 as it's now, abs as some people have REALLY old music
	d.verbose("Scaling year: {}".format(year))
	return abs(float(year/2016))

def scale_plays(plays):
	# Total plays as of 2016-06-08 16:39: 91978
	d.verbose("Scaling plays: {}".format(plays))
	return float(plays/100000)

def scale_skips(skips):
	# Total skips as of 2016-06-08 16:39: 4327
	d.verbose("Scaling skips: {}".format(skips))
	return float(skips/5000)

def scale_rating(rating):
	# Given a rating in /100 format, convert it to [0,1] range
	d.verbose("Scaling rating: {}".format(rating))
	# return float(rating/100)
	# return int(rating/20) # converts to 0 1 2 3 4 or 5
	output = np.zeros(6)
	out = int(rating/20)
	output[out] = 1
	return output

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
		if (self.start+batch_size)>len(self.data):
			self.start = 0 # reset for next epoch, I suppose?
			return # you're done! this will probably crash at the moment but oh well
		# expected return: input_feed, rating_feed
		input_feed = []
		rating_feed = []
		for i in range(batch_size):
			data_point = self.data[i+self.start]
			genre = data_point.get("Genre", "Unknown genre")
			genre = string_to_float(genre)

			year = data_point.get("Year", 0) # we'll just call it the year zero
			year = scale_year(year)

			bit_rate = data_point.get("Bit Rate", 128) # call it 128 as a default, why not
			bit_rate = scale_bit_rate(bit_rate)

			artist = data_point.get("Artist", "unknown artist")
			artist = string_to_float(artist)

			title = data_point.get("Name", "Unknown name")
			title = string_to_float(title)

			play_count = data_point.get("Play Count", 0)
			play_count = scale_plays(play_count)

			skip_count = data_point.get("Skip Count", 0)
			skip_count = scale_skips(skip_count)

			rating = data_point.get("Rating", 0)
			rating = scale_rating(rating)

			this_one = [genre, year, bit_rate, artist, title, play_count, skip_count]
			input_feed.append(this_one)
			rating_feed.append(rating)
		self.start += batch_size
		return input_feed, rating_feed

# Input handling
d.debug("Start: read plist")
tracks = plistlib.readPlist(input_data)["Tracks"]
d.debug("End: read plist")

data_set = Data_set(tracks)
d.verbose("Data set created.")
d.verbose("Data set size: {}".format(len(data_set.data)))


data_points = len(data_set.data)
data_points_train = int(data_points*0.75)
data_points_test = int(data_points*0.25)

x, y = data_set.next_batch(data_points_train) # use all the data as one batch, as Keras handles batch sizes itself
x_test, y_test = data_set.next_batch(data_points_test) # use the last quarter of the data for testing
X = []
X_test = []

# y = np.reshape(y, (len(y),1))
# y_test = np.reshape(y_test, (len(y_test),1))
y = np.vstack(y)
y_test = np.vstack(y_test)

for n in x:
	X.append(np.asarray(n))

X = np.vstack(X)
d.debug(X)

for n in x_test:
	X_test.append(np.asarray(n))

X_test = np.vstack(X_test)

d.debug("Converted to numpy ndarrays. Train points: {}. Test points: {}".format(data_points_train, data_points_test))

# Build Keras model; based on one of the tutorial ones bc why not
if not load_model:
	model = Sequential()
	model.add(Dense(64, input_dim=7 , init='uniform')) # 5-dim input: genre,year,bit_rate,artist,title, float-ified
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	model.add(Dense(6, init='uniform')) # 6 because the ratings are range(0,5)
	model.add(Activation('softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	d.debug("Model and SGD prepared.")

	model.fit(X, y, nb_epoch=50, batch_size=32)
	d.debug("Fit complete. Preparing to test.")
else:
	model = open(model_path, 'r').read()
	model = model_from_json(model)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	d.debug("Model loaded and SGD prepared.")
	if load_weights:
		model.load_weights(weights_path)
		d.debug("Weights loaded.")

model.summary()
score = model.evaluate(X_test, y_test, batch_size=16)
d.debug("")
d.debug("Test complete. Loss: {}. Accuracy: {}%".format(score[0], score[1]*100))
# Save the model and weights
json_string = model.to_json()
model.save_weights(weights_path)
open(model_path, 'w+').write(json_string)