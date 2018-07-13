#!/usr/bin/python

import os
from math import sqrt
import numpy as N
import struct

def run(args):

  #int1
  file1 = open("shci.e","rb")
  file2 = open("trusted_hci.e","rb")

  tol = float(args[2])

  index = 0
  for i in range(int(args[1])):
    calc_e = struct.unpack('d', file1.read(8))[0]
    given_e = struct.unpack('d', file2.read(8))[0]
    if abs(given_e-calc_e) > tol:
      print("\t",given_e,"-", calc_e, " > ", tol)
      print("\t FAILED Energy Test....")
    else:
      print("\t PASSED Energy Test....")
    index+=1

if __name__=="__main__":
    import sys
    run(sys.argv)
