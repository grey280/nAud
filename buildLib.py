import plistlib
import numpy				as np

import subprocess
import scipy.io.wavfile		as wav

from urllib 				import parse
import urllib

import gdebug

# Settings
input_data = "data/iTunes.plist"	# "data/iTunes.plist" the plist file to read in
output_data = "cache/data.plist"	# "cache/data.plist" the plist file to write to when done
output_directory = "cache" 			# "cache" the directory (no trailing slash) to write processed songs to
seconds_per_song = 0				# number of seconds of each song to keep; 0 for 'all'

# Tools
d = gdebug.Debugger(debug_level = 3) # 0: off 1: errors only 2: normal 3: verbose

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

def convert_song(location_path, write_path):
	# Handle the actual conversion of the song - simplifying the loop a bit
	if seconds_per_song == 0:
		d.verbose("  Using FFMPEG to convert/transfer song.")
		subprocess.run(args=["./ffmpeg", "-n", "-ac", "1", "-nostats", "-loglevel", "0", "-i", location_path, write_path])
	else:
		d.verbose("  Using FFMPEG to convert and/or shorten to {} seconds.".format(seconds_per_song))
		subprocess.run(args=["./ffmpeg", "-n","-ac", "1", "-nostats", "-loglevel", "0", "-t", "{}".format(seconds_per_song), "-i", location_path, write_path])

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
count = 0

for song_id, this_song in tracks.items():
	# Prep to process song
	d.debug("Parsing track ID: {}".format(song_id))
	location = this_song.get("Location")
	location = parse.urlparse(location)
	d.verbose("  Unquoting location path.")
	try:
		location_path = parse.unquote(location.path)
	except TypeError:
		d.error("  Type error while unquoting location path; passing unedited")
		location_path = location.path

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
	time = this_song.get("Total Time", 0)
	if genre == "Voice Memo" or genre == "Comedy":
		d.verbose("  Excluded genre. Skipping.")
		continue
	if time < 1000 * seconds_per_song: # songs that're too short cause Problems later on
		d.verbose("  Song too short. Skipping.")
		continue
	d.verbose("  Metadata prepped. Transferring.")
	if kind == "WAV audio file" or kind == "MPEG audio file" or kind == "AAC audio file":
		# convert_song(location_path, write_path)
		count += 1
		# This is SUPPOSED to be converting them to single-track audio, but it doesn't appear to be working. Annoying.
		# So either I'll have to deal with that before I do an FFT, or just... throw it all at the NN as-is?
		try:
		# 	opened = wav.read(write_path)
			new_dictionary[write_path] = track_data_to_element(this_song)
			d.verbose("  File succesfully transferred.")
			# d.debug(opened)
		except FileNotFoundError:
			d.error("  Failed to write song: not found.")
	else:
		d.verbose("  Skipping song: incompatible file format.")
# LOOP END
d.debug("Handled {} songs.".format(count))
d.verbose("Preparing to write plist to file.")
out = open(output_data, 'wb+')
plistlib.dump(new_dictionary, out)
d.verbose("  Plist write complete.")