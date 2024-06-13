import ctypes, numpy
import os
cwd = os.path.dirname(os.path.realpath(__file__))

ndpointer = numpy.ctypeslib.ndpointer
libSHCI = ctypes.cdll.LoadLibrary(cwd+'/../lib/libSHCIPostProcess.so')

readState = libSHCI.readStatec
readState.argtypes = [ctypes.c_char_p]
readState.restype = None
readState(ctypes.create_string_buffer("library call".encode("UTF-8")))
