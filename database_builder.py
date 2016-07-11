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
	"AimeeNorwich_Child_STEM_04": "vocal",
	"AlexanderRoss_GoodbyeBolero_MIX": "other",
	"AlexanderRoss_GoodbyeBolero_STEM_01": "guitar",
	"AlexanderRoss_GoodbyeBolero_STEM_02": "guitar",
	"AlexanderRoss_GoodbyeBolero_STEM_03": "guitar",
	"AlexanderRoss_GoodbyeBolero_STEM_05": "drum",
	"AlexanderRoss_GoodbyeBolero_STEM_06": "vocal",
	"AlexanderRoss_VelvetCurtain_MIX": "other",
	"AlexanderRoss_VelvetCurtain_STEM_01": "guitar",
	"AlexanderRoss_VelvetCurtain_STEM_02": "guitar",
	"AlexanderRoss_VelvetCurtain_STEM_03": "guitar",
	"AlexanderRoss_VelvetCurtain_STEM_05": "drum",
	"AlexanderRoss_VelvetCurtain_STEM_06": "vocal",
	"AlexanderRoss_VelvetCurtain_STEM_10": "guitar",
	"AmarLal_Rest_STEM_01": "guitar",
	"AmarLal_SpringDay1_STEM_01": "guitar",
	"Auctioneer_OurFutureFaces_MIX": "other",
	"Auctioneer_OurFutureFaces_STEM_04": "drum",
	"Auctioneer_OurFutureFaces_STEM_05": "guitar",
	"Auctioneer_OurFutureFaces_STEM_08": "vocal",
	"Auctioneer_OurFutureFaces_STEM_09": "drum",
	"AvaLuna_Waterduct_MIX": "other",
	"AvaLuna_Waterduct_STEM_08": "vocal",
	"AvaLuna_Waterduct_STEM_04": "drum"
}

database = {}

with open(working_file, "r") as infile:
	database = json.load(infile)

database.update(songs_to_add)

with open(working_file, "w") as outfile:
	json.dump(database, outfile, sort_keys=True, indent=4, separators=(',',': '))