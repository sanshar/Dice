#!/usr/bin/python                                                                                                                                             \
                                                                                                                                                               

import os
from math import sqrt
import numpy as N

def run(file1, file2, tol):

  #int1                                                                                                                                                       \
                                                                                                                                                               
  filer1 = open(file1,"r")
  filer2 = open(file2,"r")
  sz = int(filer1.readline().split()[0])
  sz2 = int(filer2.readline().split()[0])
  mat1 = N.zeros((sz,sz))
  mat2 = N.zeros((sz2,sz2))
  for line in filer1.readlines():
    linesp = line.split()
    mat1[int(linesp[0]),int(linesp[1])] = float(linesp[2])
  for line in filer2.readlines():
    linesp = line.split()
    mat2[int(linesp[0]),int(linesp[1])] = float(linesp[2])
  filer1.close()
  filer2.close()
  val = 0.
  for i in xrange(0,sz):
    for j in xrange(0,sz):
      res = (mat1[i,j] - mat2[i,j])*(mat1[i,j] - mat2[i,j])
      #res = (mat1[i,j] - mat2[j,i])*(mat1[i,j] - mat2[j,i])
      val = val + res
  if val > float(tol):
    print "FAILED ...."
  else:
    print "PASSED ...."

if __name__=="__main__":
    import sys
    run(sys.argv[1], sys.argv[2], sys.argv[3])

