import plistlib

data = plistlib.readPlist("./data/iTunes.plist")

print(data["Tracks"])