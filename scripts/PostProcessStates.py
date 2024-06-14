import ctypes, numpy
import os
from mpi4py import MPI
cwd = os.path.dirname(os.path.realpath(__file__))

ndpointer = numpy.ctypeslib.ndpointer
libSHCI = ctypes.cdll.LoadLibrary(cwd+'/../lib/libSHCIPostProcess.so')

transitionRDMc = libSHCI.transitionRDMc
transitionRDMc.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
transitionRDMc.restype = None



def makeTransitionRDM(fname1, fname2, norb, nelec):
    fname1 = ctypes.create_string_buffer(fname1.encode("UTF-8"))
    fname2 = ctypes.create_string_buffer(fname2.encode("UTF-8"))

    transitionRDMc(fname1, fname2, norb, nelec)

