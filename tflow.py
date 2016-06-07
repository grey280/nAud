import plistlib
# import tensorflow	as tf

# Helper Functions
def string_to_float(original_name):
	# Convert a string to a float: not a *strict* mapping algorithm, but probably close enough to work
	multiplier = 0.01
	outvar = 0.1
	for c in original_name:
		outvar = outvar + (multiplier * (ord(c) - 70))
		multiplier = multiplier * 0.1
	return abs(outvar)

def scale_bit_rate(bit_rate):
	# Scale bitrate into a float; using 1500 as the peak value bc 1411 is the highest in my library
	return float(bit_rate/1500)

def scale_year(year):
	# Scale year into a float; using 2016 as it's now, abs as some people have REALLY old music
	return abs(float(year/2016))


# Read in data to process
data = plistlib.readPlist("./data/iTunes.plist")
tracks = data["Tracks"]			# extricate the only part we actually care about



# [x] There are 55 genres in my iTunes library, use that as the scale (alphabetical order?)
#		Just gonna use the same function as I'm using for artist/title
# [x] Years can be edited assuming they range 0-2016, scale down to [0,1]
# [x] Bit rate can be scaled from 0kbps to 1500 kbps; one thing has 8 kbps.
#		Might need a boolean switch for variable bitrate
# [x] For artists, write a function that maps the alphabet?
# [x] Do the same for the song title

# Ratings go from 0 to 5, inclusive