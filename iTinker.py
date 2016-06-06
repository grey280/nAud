import plistlib

# So, fun thing about the "iTunes Library Data.xml" file: it's not a *real* XML implementation.
# It's actually a godawful combination of an Apple-specific Property List (.plist) file, crammed
# into something that, at first glance, *looks* like XML. Let me tell you: it isn't.
# Reading it has to be done with `plistlib` rather than the `XML` set of packages, beacuse if you
# try to parse it with the XML stuff you'll wind up tearing out your hair.

# (The fact that Apple is using such a *weird* way of storing data for iTunes is really just par
#  for the course, with iTunes.)

data = plistlib.readPlist("./data/iTunes.plist")
tracks = data["Tracks"]			# extricate the only part we actually care about

for track, data in tracks.items():
	if data.get("Year", 0)==2016: # have to use .get with default, because if there's no data then it'll just crash
		print(data)