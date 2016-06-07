import plistlib

# So, fun thing about the "iTunes Library Data.xml" file: it's not a *real* XML implementation.
# It's actually a godawful combination of an Apple-specific Property List (.plist) file, crammed
# into something that, at first glance, *looks* like XML. Let me tell you: it isn't.
# Reading it has to be done with `plistlib` rather than the `XML` set of packages, beacuse if you
# try to parse it with the XML stuff you'll wind up tearing out your hair.

# (The fact that Apple is using such a *weird* way of storing data for iTunes is really just par
#  for the course, with iTunes.)

# Helper functions
def list_kinds(dataSet):
	kinds = set()
	for track, data in dataSet.items():
		kinds.add(data.get("Kind", "Unknown kind"))

	return kinds

def print_track_metadata(track, data):
	# have to use .get with default, because if there's no data then it'll just crash
	kind = data.get("Kind", "Unknown kind")
	if kind.endswith("audio file"):			# Only print songs - videos and stuff are mixed in with the raw data
		genre = data.get("Genre", "unknown")
		year = data.get("Year", "unknown")
		bit_rate = data.get("Bit Rate", "unknown")
		artist = data.get("Artist", "unknown artist")
		name = data.get("Name", "Unknown name")
		rating = data.get("Rating", 0)
		rating = int(rating/20)
		print("{} ({}) by {}. {}. {} kbps. {}/5".format(name, year, artist, genre, bit_rate, rating))

def print_track_metadata_CSV(track, data):
	kind = data.get("Kind", "Unknown kind")
	if kind.endswith("audio file"):
		genre = data.get("Genre", "unknown")
		year = data.get("Year", "unknown")
		bit_rate = data.get("Bit Rate", "unknown")
		artist = data.get("Artist", "unknown artist")
		name = data.get("Name", "Unknown name")
		rating = data.get("Rating", 0)
		rating = int(rating/20)
		print("{},{},{},{},{},{}".format(name, year, artist, genre, bit_rate, rating))


# Import data
data = plistlib.readPlist("./data/iTunes.plist")
tracks = data["Tracks"]			# extricate the only part we actually care about

# Print data
for track, data in tracks.items():
	print("name,year,artist,genre,bit_rate,rating")
	print_track_metadata_CSV(track, data)