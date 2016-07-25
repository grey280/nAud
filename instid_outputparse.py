# instid_outputparse: Takes a JSON array-of-arrays input, converts it to a dictionary and saves that as JSON

import json

working_file = "output/raw.json"
output_file = "output/parsed.json"

inputs = []
outputs = []

def convert_kind(kind):
	# Helper function - converts a string kind to a one-hot
	out = []
	if kind == "drum":
		out= [0, 1, 0, 0]
	elif kind == "guitar":
		out= [0, 0, 1, 0]
	elif kind == "vocal":
		out= [0, 0, 0, 1]
	else:
		out= [1, 0, 0, 0]
	output = np.asarray(out).reshape(1,4)
	return output

def revert_kind(kind):
	# Helper funciton - converts a one-hot kind to a string
	out = ""
	if kind == 0:
		out = "other"
	elif kind == 1:
		out = "drum"
	elif kind == 2:
		out = "guitar"
	else:
		out = "vocal"
	return out

def one_hot_to_int(one_hot):
	# Helper function - converts a one-hot to an integer
	currentMax = 0.0
	currentMaxId = 0
	for i in range(len(one_hot)):
		if one_hot[i] > currentMax:
			currentMax = one_hot[i]
			currentMaxId = i
	return currentMaxId

# Load input
temp_inputs = []
with open(working_file, "r") as infile:
	temp_inputs = json.load(infile)

inputs = [one_hot_to_int(num) for num in temp_inputs] # I *think* this will work?


currentKind = 5
currentStart = 0
for i in range(len(inputs)):
	if inputs[i] != currentKind:
		this_one = {"start":currentStart, "stop":i, "category": revert_kind(inputs[i])}
		outputs.append(this_one)
		currentKind = inputs[i]
		currentStart = i


# Save output
with open(working_file, "w") as outfile:
	json.dump(database, outfile, sort_keys=True, indent=4, separators=(',',': '))