import json

working_file = "database.json"
# Song format: "file_name": "kind"
# 	example provided
songs_to_add={
	# "LizNelson_Rainfall_STEM_04": "guitar"
}

database = {}

with open(working_file, "r") as infile:
	database = json.load(infile)

database.update(songs_to_add)

with open(working_file, "w") as outfile:
	json.dump(database, outfile, sort_keys=True, indent=4, separators=(',',': '))