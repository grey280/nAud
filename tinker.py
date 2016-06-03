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
		self.localPrint("############### {}".format(message))
		self.localPrint("Type: {}".format(type(printObj)))
		self.localPrint("Item Types: {}".format(type(printObj[0])))
		self.localPrint("Size: {}".format(len(printObj)))
		self.localPrint("{}".format(printObj))

	def parsePrintArray(self, printObj):
		# Prints array for parsing, in Excel or something
		# time real imaginary magnitude, space-delineated, with those headings at the top
		self.localPrint("time real imaginary magnitude")
		for a in range(0, len(printObj)):
			self.localPrint("{} {} {} {}".format(a, printObj[a].real, printObj[a].imag, np.sqrt((printObj[a].real*printObj[a].real)+(printObj[a].imag*printObj[a].imag))))

	def localPrint(self, msg):
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

def writeSample(name, data, sampleRate=44100):
	temp = "output/{}.wav".format(name)
	wav.write(temp, sampleRate, data)

# Doing my tinkering
parser = argparse.ArgumentParser(description='Run an FFT on a sample.')
parser.add_argument('i', metavar='I', nargs=1, type=str, default="440hz", help='name of sample to process')
parser.add_argument('-o', nargs='?', type=argparse.FileType('w'), default=sys.stdout, help='file to output results to')
args = parser.parse_args()

dbug = debugger(args.o)
srcData = readSample(args.i[0])

fftD = np.fft.fft(srcData[1], int(44100/2))

dbug.parsePrintArray(fftD)