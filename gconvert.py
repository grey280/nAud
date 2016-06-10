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

# Genre parsing
number_of_genres = 13
genre_labels = {
	"Unknown": 0,
	"Pop": 1,
	"Rock": 2,
	"Alternative": 3,
	"Indie": 4,
	"Soundtrack": 5,
	"Rap": 6,
	"Classical": 7,
	"Electronic": 8,
	"Holiday": 9,
	"Country": 10,
	"Ambient": 11,
	"Other": 12
}
def convert_genre(genre):
	genres = {
		"Unknown": 0,
		"Acoustic": 1,
		"Adult Alternative": 2,
		"Alternative": 3,
		"Alternative Pop": 4,
		"Alternative Punk": 5,
		"Alternative Rock": 6,
		"Ambient": 7,
		"Bluegrass": 8,
		"Children's Music": 9,
		"Christian & Gospel": 10,
		"Classical": 11,
		"Comedy": 12,
		"Country": 13,
		"Dance": 14,
		"Disco": 15,
		"Dubstep": 16,
		"Electronic": 17,
		"Folk": 18,
		"Hip Hop/Rap": 19,
		"Holiday": 20,
		"House": 21,
		"Indie": 22,
		"Indie Alternative": 23,
		"Indie Electronic": 24,
		"Indie Folk": 25,
		"Indie Pop": 26,
		"Indie Rock": 27,
		"Instrumental": 28,
		"Jazz": 29,
		"K-Pop": 30,
		"Latin": 31,
		"Mashup": 32,
		"Modern": 33,
		"Modern Classical": 34,
		"Pop": 35,
		"Pop Punk": 36,
		"Pop Rock": 37,
		"Post Rock": 38,
		"Post-Punk": 39,
		"Punk": 40,
		"Punk Rock": 41,
		"R&B": 42,
		"R&B/Pop": 43,
		"R&B/Soul": 44,
		"Rap": 45,
		"Remix": 46,
		"Rock": 47,
		"Rock/Pop": 48,
		"Score": 49,
		"Shoegaze": 50,
		"Singer/Songwriter": 51,
		"Soundtrack": 52,
		"Synthpop": 53,
		"Trance": 54
	}
	genres_convert = [0, 3, 3, 3, 3, 3, 3, 11, 10, 12, 12, 7, 12, 10, 8, 8, 8, 8, 10, 6, 9, 8, 4, 4, 4, 4, 4, 4, 7, 12, 1, 12, 12, 7, 7, 1, 1, 1, 11, 2, 2, 2, 6, 6, 6, 6, 12, 2, 2, 7, 11, 4, 7, 1, 11]
	temp = genres.get(genre, 0)
	return genres_convert[temp]
	# return float(float(genres_convert[temp])/number_of_genres) # I *really* want it to be a float, okay

def genre_to_label(genre):
	titles=["Unknown", "Pop", "Rock", "Alternative", "Indie", "Soundtrack", "Rap", "Classical", "Electronic", "Holiday", "Country", "Ambient", "Other"]
	if genre > len(titles)-1:
		return titles[0]
	return titles[genre]

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