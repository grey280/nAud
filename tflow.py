import plistlib
import tensorflow	as tf

# Import data
data = plistlib.readPlist("./data/iTunes.plist")
tracks = data["Tracks"]			# extricate the only part we actually care about