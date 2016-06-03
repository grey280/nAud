import numpy				as np
import scipy.io.wavfile		as wav

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



srcData = wav.read("samples/440hz.wav")

fftD = np.fft.fft(srcData[1], int(44100/2))

altPrintArray(fftD)

scaled = np.int16(fftD.real/np.max(np.abs(fftD.real)) * 32767)

# wav.write("output.wav", srcData[0], scaled)