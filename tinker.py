import numpy				as np
import scipy.io.wavfile		as wav


# Helper Functions
def debugPrintArray(printObj, message="Debug Print: "):
	print(message)
	print("		Type: {}".format(type(printObj)))
	print("		Item Types: {}".format(type(printObj[0])))
	print("		Size: {}".format(len(printObj)))
	print("		{}".format(printObj))

def parsePrintArray(printObj):
	for a in printObj:
		print("{} {}".format(a.real, a.imag))

def altPrintArray(printObj):
	print("time real imaginary magnitude")
	for a in range(0, len(printObj)):
		print("{} {} {} {}".format(a, printObj[a].real, printObj[a].imag, np.sqrt((printObj[a].real*printObj[a].real)+(printObj[a].imag*printObj[a].imag))))

def readSample(name):
	temp = "samples/{}.wav".format(name)
	return wav.read(temp)

def scaleData(data):
	return np.int16(data.real/np.max(np.abs(data.real)) * 32767)

def writeSample(name, data, sampleRate=44100):
	temp = "output/{}.wav".format(name)
	wav.write(temp, sampleRate, data)

# Doing my tinkering
srcData = readSample("440hz")

fftD = np.fft.fft(srcData[1], int(44100/2))

altPrintArray(fftD)

scaled = scaleData(fftD)

writeSample("output", scaled, srcData[0])