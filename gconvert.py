import hashlib
import numpy			as np

def string_to_int(original_name):
	# uses a hash function to convert, no weird scaling stuff
	temp = int(hashlib.sha1(original_name.encode('utf-8')).hexdigest(), 16) % (2**14)
	return int(temp)

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
	try: 
		return abs(float(year/2016))
	except TypeError:
		return 1.0 # we'll just call it 2016 if it's a string or something

def scale_plays(plays):
	# Total plays as of 2016-06-08 16:39: 91978
	return float(plays/100000)

def scale_skips(skips):
	# Total skips as of 2016-06-08 16:39: 4327
	return float(skips/5000)

def scale_rating(rating):
	# Given a rating in /100 format, convert it to [0,1] range
	output = np.zeros(6, dtype=int)
	out = int(rating/20)
	output[out] = 1
	return output

def scale_genre(genre):
	output = np.zeros(number_of_genres, dtype=int)
	output[genre] = 1
	return output

number_of_genres = 4
meta_genres = {
		"Other": 0,
		"Art": 1,
		"Pop": 2,
		"Traditional": 3
	}

def int_to_one_hot(input_val, oh_size=1024):
	out = []
	for i in range(oh_size):
		if (i+1)==input_val:
			out.append(1)
		else:
			out.append(0)
	return out

def convert_genre(genre):
	genres = {
		"Unknown": "Other",
		"Acoustic": "Pop",
		"Adult Alternative": "Pop",
		"Alternative": "Pop",
		"Alternative Pop": "Pop",
		"Alternative Punk": "Pop",
		"Alternative Rock": "Pop",
		"Ambient":"Pop",
		"Bluegrass":"Traditional",
		"Children's Music":"Pop",
		"Christian & Gospel": "Art",
		"Classical": "Art",
		"Comedy": "Other",
		"Country": "Pop",
		"Dance": "Pop",
		"Disco": "Pop",
		"Dubstep": "Pop",
		"Electronic": "Pop",
		"Folk": "Traditional",
		"Hip Hop/Rap": "Pop",
		"Holiday": "Pop",
		"House": "Pop",
		"Indie": "Pop",
		"Indie Alternative": "Pop",
		"Indie Electronic": "Pop",
		"Indie Folk": "Traditional",
		"Indie Pop": "Pop",
		"Indie Rock": "Pop",
		"Instrumental": "Art",
		"Jazz": "Pop",
		"K-Pop": "Pop",
		"Latin": "Pop",
		"Mashup": "Pop",
		"Modern": "Art",
		"Modern Classical": "Art",
		"Pop": "Pop",
		"Pop Punk": "Pop",
		"Pop Rock": "Pop",
		"Post Rock": "Pop",
		"Post-Punk": "Pop",
		"Punk": "Pop",
		"Punk Rock": "Pop",
		"R&B": "Pop",
		"R&B/Pop": "Pop",
		"R&B/Soul": "Pop",
		"Rap": "Pop",
		"Remix": "Pop",
		"Rock": "Pop",
		"Rock/Pop": "Pop",
		"Score": "Art",
		"Shoegaze": "Pop",
		"Singer/Songwriter": "Pop",
		"Soundtrack": "Pop",
		"Synthpop": "Pop",
		"Trance": "Pop",
		"World": "Traditional"
	}
	temp = genres.get(genre, "Other")
	return meta_genres.get(temp, 0)

def number_to_label(genre_id):
	meta_genres = ["Other", "Art", "Pop", "Traditional"]
	if genre_id > len(meta_genres):
		return meta_genres[0]
	return meta_genres[genre_id]

def descale_genre(genre):
	return int(genre*number_of_genres)

def one_hot_to_int(one_hot):
	currentMax = 0.0
	currentMaxId = 0
	for i in range(len(one_hot)):
		if one_hot[i] > currentMax:
			currentMax = one_hot[i]
			currentMaxId = i
	return currentMaxId