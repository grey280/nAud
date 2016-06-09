import plistlib
import gdebug
import numpy		as np
import subprocess

d = gdebug.Debugger(debug_level = 2) # 0: off 1: errors only 2: normal 3: verbose
input_data = "data/iTunes.plist"

d.verbose("Preparing to read tracks in.")
tracks = plistlib.readPlist(input_data)["Tracks"]
d.debug("Tracks read.")

# The kinds of files iTunes has: 'Matched AAC audio file', 'Protected MPEG-4 video file', 'MPEG audio file', 'MPEG-4 video file', 'unknown kind', 'WAV audio file', 'AAC audio file', 'Purchased AAC audio file', 'Purchased MPEG-4 video file', 'Protected AAC audio file'
# Kinds we want to process: MPEG audio file, AAC audio file
# Kinds we want to copy without processing:  WAV audio file

