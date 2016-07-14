import json

working_file = "output/raw.json"
output_file = "output/parsed.json"

inputs = []
outputs = {}

def convert_kind(kind):
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
	out = ""
	if kind[0] == 1:
		out = "other"
	elif kind[1] == 1:
		out = "drum"
	elif kind[2] == 1:
		out = "guitar"
	else:
		out = "vocal"
	return out

# Load input
with open(working_file, "r") as infile:
	inputs = json.load(infile)

currentKind = 5
currentStart = 0
currentEnd = 0
for i in range(len(inputs)):
	if inputs[i] != currentKind:
		




# Save output
with open(working_file, "w") as outfile:
	json.dump(database, outfile, sort_keys=True, indent=4, separators=(',',': '))