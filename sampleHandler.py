import plistlib
import gdebug
import numpy		as np
import subprocess

import random
from urllib import parse
import urllib

d = gdebug.Debugger(debug_level = 2) # 0: off 1: errors only 2: normal 3: verbose
input_data = "data/iTunes.plist"

d.verbose("Preparing to read tracks in.")
tracks = plistlib.readPlist(input_data)["Tracks"]
d.debug("Tracks read.")

# The kinds of files iTunes has: 'Matched AAC audio file', 'Protected MPEG-4 video file', 'MPEG audio file', 'MPEG-4 video file', 'unknown kind', 'WAV audio file', 'AAC audio file', 'Purchased AAC audio file', 'Purchased MPEG-4 video file', 'Protected AAC audio file'
# Kinds we want to process: MPEG audio file, AAC audio file
# Kinds we want to copy without processing:  WAV audio file

# Directory where we'll put things: "cache"
# File name format: "year.artist.title.wav"
# 		This format isn't *super great* for the rating prediction stuff, but it's what we'll have
#		for the genre-guessing stuff, which I'm hoping to be able to reuse this code for. So.


this_song = random.choice(list(tracks.keys())) # picks a random track to look at
					# eventually that'll be replaced by a for loop going through all of them

# Prep to process song
location = this_song.get("Location")
location = parse.urlparse(location)
location_path = parse.unquote(location.path)
year = this_song.get("Year", 2016)
artist = this_song.get("Artist", "unknown")
title = this_song.get("Name","unknown")
write_path = "{}.{}.{}.wav".format(year,artist,title)
kind = this_song.get("Kind", "unknown kind")

if kind == "WAV audio file":
	subprocess.run(args=["cp", location_path, write_path])
elif kind == "MPEG audio file" || kind == "AAC audio file":
	subprocess.run(args=["./ffmpeg", "-ac", "1", "-i", location_path, write_path])
	# Fun story, that's *supposed* to be converting it to mono, but I don't know if it actually does
	# On the other hand, the WAV audio files we get from the other one *also* won't be mono, so
	# maybe I'll just write the AI input to deal with that. Or ignore the second track.
# no else, because we don't care beyond that - those ones get thrown out


d.debug(this_song)
track = tracks.get(this_song)
d.debug(track)
location = track.get("Location")
d.debug(location)
location = parse.urlparse(location).path
d.debug(location)
location = parse.unquote(location)
d.debug(location)
# opened = wav.read(location)
outputfile = "cache/{}.wav".format(time.strftime("%d.%m.%Y.%H.%M.%S"))
subprocess.run(args=["./ffmpeg", "-ac", "1", "-i", location, outputfile])
opened = wav.read(outputfile)

d.debug(opened)