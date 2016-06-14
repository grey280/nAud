# gDebug: my own local debugging stuff, so I can be slightly fancier about this
import sys

class Debugger:
	def __init__(self, debug_level=0):
		self.debug_level = debug_level

	def debug_level():
	    doc = "Debug level: 0: none, 1: errors only, 2: normal, 3: verbose"
	    def fget(self):
	        return self._debug_level
	    def fset(self, value):
	        self._debug_level = value
	    def fdel(self):
	        del self._debug_level
	    return locals()
	debug_level = property(**debug_level())

	def error(self, string):
		if self.debug_level > 0:
			print(string)

	def debug(self, string):
		if self.debug_level > 1:
			print(string)

	def verbose(self, string):
		if self.debug_level > 2:
			print(string)
	def progress(self, message, currentval, maxval):
		if self.debug_level > 1:
			pbar = self._progress_bar_(currentval, maxval)
			if not currentval == maxval:
				sys.stdout.write("\r{}\n{} {}/{}".format(message, pbar, currentval, maxval))
			else:
				sys.stdout.write("\r{}\n{} complete\n".format(message, pbar))

	def _progress_bar_(self, currentval, maxval, length=20):
		output = ""
		for i in range(length):
			if int(length*currentval/maxval) > i:
				output = "{}=".format(output)
			else:
				output = "{} ".format(output)
		return "[{}]".format(output)
