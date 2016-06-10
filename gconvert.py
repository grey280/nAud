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