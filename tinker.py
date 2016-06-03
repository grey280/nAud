import numpy				as np
import scipy.io.wavfile		as wav
import argparse
import sys


class debugger():
	def __init__(self, printFile=None):
		if(printFile!=None):
			self.printFile = printFile
		else:
			self.printFile = None

	def debugPrintArray(self, printObj, message="Debug: "):
		# Prints a neatly-formatted readout in style:
		# ############### (message)
		# Type: (type information)
		# Item Types: (type of array items)
		# Size: (size of array)
		# (array)
		self.prnt("############### {}".format(message))
		self.prnt("Type: {}".format(type(printObj)))
		self.prnt("Item Types: {}".format(type(printObj[0])))
		self.prnt("Size: {}".format(len(printObj)))
		self.prnt("{}".format(printObj))

	def parsePrintArray(self, printObj):
		# Prints array for parsing, in Excel or something
		# time real imaginary magnitude, space-delineated, with those headings at the top
		self.prnt("Time,Real Component,Imaginary Component,Magnitude")
		for a in range(0, len(printObj)):
			self.prnt("{},{},{},{}".format(a, printObj[a].real, printObj[a].imag, np.sqrt((printObj[a].real*printObj[a].real)+(printObj[a].imag*printObj[a].imag))))

	def prnt(self, msg):
		if(self.printFile!=None):
			print(msg, file=self.printFile)
		else:
			print(msg)

# Helper functions
def readSample(name):
	temp = "samples/{}.wav".format(name)
	return wav.read(temp)

def scaleData(data):
	# Scales data for output via scipy.io.wavfile: converts real components to integers
	return np.int16(data.real/np.max(np.abs(data.real)) * 32767)

# Load variables and whatnot
parser = argparse.ArgumentParser(description='Run an FFT on a sample.')
parser.add_argument('i', metavar='I', nargs=1, type=str, default="440hz", help='name of sample to process (reads samples/[I].wav')
parser.add_argument('-l', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='file to write logs to')
args = parser.parse_args()
dbug = debugger(args.l)
srcData = readSample(args.i[0])

# Parse some stuff!
fftD = np.fft.fft(srcData[1], int(44100/2))

dbug.parsePrintArray(fftD)