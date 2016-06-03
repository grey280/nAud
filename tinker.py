import numpy				as np
import scipy.io.wavfile		as wav


class debugger():
	def debugPrintArray(self, printObj, message="Debug: "):
		# Prints a neatly-formatted readout in style:
		# ############### (message)
		# Type: (type information)
		# Item Types: (type of array items)
		# Size: (size of array)
		# (array)
		print("############### {}".format(message))
		print("Type: {}".format(type(printObj)))
		print("Item Types: {}".format(type(printObj[0])))
		print("Size: {}".format(len(printObj)))
		print("{}".format(printObj))

	def parsePrintArray(self, printObj):
		# Prints array for parsing, in Excel or something
		# time real imaginary magnitude, space-delineated, with those headings at the top
		print("time real imaginary magnitude")
		for a in range(0, len(printObj)):
			print("{} {} {} {}".format(a, printObj[a].real, printObj[a].imag, np.sqrt((printObj[a].real*printObj[a].real)+(printObj[a].imag*printObj[a].imag))))

# Helper functions
def readSample(name):
	temp = "samples/{}.wav".format(name)
	return wav.read(temp)

def scaleData(data):
	# Scales data for output via scipy.io.wavfile: converts real components to integers
	return np.int16(data.real/np.max(np.abs(data.real)) * 32767)

def writeSample(name, data, sampleRate=44100):
	temp = "output/{}.wav".format(name)
	wav.write(temp, sampleRate, data)

# Doing my tinkering
dbug = debugger()
srcData = readSample("440hz")

fftD = np.fft.fft(srcData[1], int(44100/2))

dbug.parsePrintArray(fftD)

scaled = scaleData(fftD)

# writeSample("output", scaled, srcData[0])