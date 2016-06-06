import plistlib

data = plistlib.readPlist("./data/iTunes.plist")

tracks = data["Tracks"]

for track, data in tracks.items():
	if data.get("Year", 0)==2016:
		# print(data['Title'])
		print(data)
	# if tracks[track]["Year"]==2016:
	# print(tracks[track])
	# print(tracks[track])
	# print("{}: {}".format(track, data['Year']))