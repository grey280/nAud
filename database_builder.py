import json

working_file = "data/database.json"
# Song format: "file_name": "kind"
# 	example provided
songs_to_add={
	# "LizNelson_Rainfall_STEM_04": "guitar"
	"AClassicEducation_NightOwl_STEM_02": "drum",
	"AClassicEducation_NightOwl_STEM_05": "guitar",
	"AClassicEducation_NightOwl_STEM_08": "vocal",
	"AClassicEducation_NightOwl_MIX": "other",
	"AimeeNorwich_Child_STEM_01": "drum",
	"AimeeNorwich_Child_STEM_03": "guitar",
	"AimeeNorwich_Child_MIX": "other",
	"AimeeNorwich_Child_STEM_07": "guitar",
	"AimeeNorwich_Child_STEM_04": "vocal"
}

database = {}

with open(working_file, "r") as infile:
	database = json.load(infile)

database.update(songs_to_add)

with open(working_file, "w") as outfile:
	json.dump(database, outfile, sort_keys=True, indent=4, separators=(',',': '))