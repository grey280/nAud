import plistlib
import numpy				as np
import subprocess
import scipy.io.wavfile		as wav
import random # testing only
from urllib import parse
import urllib
import gdebug

# Global stuff
d = gdebug.Debugger(debug_level = 3) # 0: off 1: errors only 2: normal 3: verbose
input_data = "data/iTunes.plist"
output_data = "cache/data.plist"
output_directory = "cache" # no trailing slash, script adds that

# Helper functions
def track_data_to_element(data):
	# Converts the 'data' chunk of a track to a nice dictionary for writing to our
	#	own plist file later.
	genre = data.get("Genre", "unknown")
	year = data.get("Year", 2016)
	bit_rate = data.get("Bit Rate", 128)
	artist = data.get("Artist", "unknown")
	title = data.get("Name", "unknown")
	play_count = data.get("Play Count", 0)
	skip_count = data.get("Skip Count", 0)
	rating = data.get("Rating", 0)
	return {'year': year, 'artist': artist, 'title': title, 'genre': genre, 'bit_rate': bit_rate, 'play_count': play_count, 'skip_count': skip_count, 'rating': rating}



d.verbose("Preparing to read tracks in.")
tracks = plistlib.readPlist(input_data)["Tracks"]
d.debug("Track database loaded.")

# We're going to store all of the metadata to a DIFFERENT plist file, since we're cutting out a bunch of files that aren't compatible.
new_dictionary = {}

# The kinds of files iTunes has: 'Matched AAC audio file', 'Protected MPEG-4 video file', 'MPEG audio file', 'MPEG-4 video file', 'unknown kind', 'WAV audio file', 'AAC audio file', 'Purchased AAC audio file', 'Purchased MPEG-4 video file', 'Protected AAC audio file'
# Kinds we want to process: MPEG audio file, AAC audio file
# Kinds we want to copy without processing:  WAV audio file

# Directory where we'll put things: "cache"
# File name format: "year.artist.title.wav"
# 		This format isn't *super great* for the rating prediction stuff, but it's what we'll have
#		for the genre-guessing stuff, which I'm hoping to be able to reuse this code for. So.


# LOOP START
# this_song = random.choice(list(tracks.keys())) # picks a random track to look at
# this_song = tracks.get(this_song) # these two lines were for testing before I implemented the loop
for song_id, this_song in tracks.items():
	# Prep to process song
	d.verbose("Parsing track.")
	location = this_song.get("Location")
	location = parse.urlparse(location)
	d.verbose("  Unquoting location path.")
	try:
		location_path = parse.unquote(location.path)
	except TypeError:
		d.error("  Type error while unquoting location path; passing unedited")
		location_path = location.path

	# if "%" in location.path:
	# 	d.verbose("  Unquoting track location.")
	# 	location_path = parse.unquote(location.path)
	# else:
	# 	d.verbose("  Track doesn't need unquoting.")
	# 	location_path = location.path
	year = this_song.get("Year", 2016)
	artist = this_song.get("Artist", "unknown")
	artist = artist.replace('/', '')
	title = this_song.get("Name","unknown")
	title = title.replace('/', '') # make the output file name safe
	album = this_song.get("Album", "unknown")
	album = album.replace('/','')
	write_path = "{}/{}.{}.{}.{}.wav".format(output_directory, year,artist,album,title)
	kind = this_song.get("Kind", "unknown kind")
	genre = this_song.get("Genre", "unknown")
	if genre == "Voice Memo":
		continue
	d.verbose("  Metadata prepped. Transferring.")
	if kind == "WAV audio file" or kind == "MPEG audio file" or kind == "AAC audio file":
		d.verbose("  Using FFMPEG to convert and/or shorten to 20 seconds.")
		subprocess.run(args=["./ffmpeg", "-ac", "1", "-t", "10", "-i", location_path, write_path])
		# This is SUPPOSED to be converting them to single-track audio, but it doesn't appear to be working. Annoying.
		# So either I'll have to deal with that before I do an FFT, or just... throw it all at the NN as-is?
		try:
			opened = wav.read(write_path)
			new_dictionary[write_path] = track_data_to_element(this_song)
			d.verbose("  File succesfully transferred.")
			d.debug(opened)
		except FileNotFoundError:
			d.error("  Failed to write song: not found.")
	else:
		d.verbose("  Skipping song: incompatible file format.")
# LOOP END

d.verbose("Preparing to dump plist to file.")
out = open(output_data, 'wb+')
plistlib.dump(new_dictionary, out)
d.verbose("  Plist dump complete.")