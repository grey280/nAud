# instid_database_builder: Database building utility for instrument identification. Any song in the songs_to_add dictionary is added to the database file indicated.

import json

working_file = "data/database.json"
# Song format: "file_name": "kind"
#   available kinds: guitar, vocal, drum, other
songs_to_add={
	# "LizNelson_Rainfall_STEM_04": "guitar" # example input
}

database = {}

with open(working_file, "r") as infile:
	database = json.load(infile)

database.update(songs_to_add)

with open(working_file, "w") as outfile:
	json.dump(database, outfile, sort_keys=True, indent=4, separators=(',',': '))