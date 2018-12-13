import os
from math import sqrt
import numpy as np
import numpy.testing as npt
import struct


# Smoke tests for the smaller problems
def test_energy(tol=1e-6):

    #int1
    file1 = open("shci.e", "rb")
    file2 = open("trusted_hci.e", "rb")

    # tol = float(args[2])

    index = 0
    for i in range(1):
        calc_e = struct.unpack('d', file1.read(8))[0]
        given_e = struct.unpack('d', file2.read(8))[0]
        # if abs(given_e-calc_e) > tol:
        #   print("\t",given_e,"-", calc_e, " > ", tol)
        #   print("\t FAILED Energy Test....")
        # else:
        #   print("\t PASSED Energy Test....")
        assert (abs(given_e - calc_e) < tol)
        index += 1


def compare_1rdm(file1="spatialRDM.0.0.txt", file2="trusted2RDM.txt",
                 tol=1e-8):

    filer1 = open(file1, "r")
    filer2 = open(file2, "r")
    sz = int(filer1.readline().split()[0])
    sz2 = int(filer2.readline().split()[0])
    mat1 = np.zeros((sz, sz))
    mat2 = np.zeros((sz2, sz2))
    for line in filer1.readlines():
        linesp = line.split()
        mat1[int(linesp[0]), int(linesp[1])] = float(linesp[2])
    for line in filer2.readlines():
        linesp = line.split()
        mat2[int(linesp[0]), int(linesp[1])] = float(linesp[2])
    filer1.close()
    filer2.close()
    val = 0.
    for i in range(0, sz):
        for j in range(0, sz):
            res = (mat1[i, j] - mat2[i, j]) * (mat1[i, j] - mat2[i, j])
            #res = (mat1[i,j] - mat2[j,i])*(mat1[i,j] - mat2[j,i])
            val = val + res
    # if val > float(tol):
    #     print("FAILED ....")
    # else:
    #     print("PASSED ....")
    assert (val < float(tol))


def compare_2rdm(file1="spatialRDM.0.0.txt", file2="trusted2RDM.txt",
                 tol=1e-8):

    #int1
    filer1 = open(file1, "r")
    filer2 = open(file2, "r")
    sz = int(filer1.readline().split()[0])
    sz2 = int(filer2.readline().split()[0])
    mat1 = np.zeros((sz, sz, sz, sz))
    mat2 = np.zeros((sz2, sz2, sz2, sz2))
    for line in filer1.readlines():
        linesp = line.split()
        mat1[int(linesp[0]),
             int(linesp[1]),
             int(linesp[2]),
             int(linesp[3])] = float(linesp[4])
    for line in filer2.readlines():
        linesp = line.split()
        mat2[int(linesp[0]),
             int(linesp[1]),
             int(linesp[2]),
             int(linesp[3])] = float(linesp[4])
    filer1.close()
    filer2.close()
    val = 0.
    for i in range(0, sz):
        for j in range(0, sz):
            for k in range(0, sz):
                for l in range(0, sz):
                    res = (mat1[i, j, k, l] - mat2[i, j, k, l]) * (
                        mat1[i, j, k, l] - mat2[i, j, k, l])
                    val = val + res
    # if val > float(tol):
    #     print("\t ", val, " > ", tol)
    #     print("\t FAILED 2RDM....")
    # else:
    #     print("\t PASSED 2RDM....")
    assert (val < float(tol))


def cleanup():
    # fn := filename
    for fn in os.listdir('.'):
        if fn.endswith('.bkp') or fn.endswith("0.txt") or fn == "shci.e" or fn == "output.dat":
            # print("Removing %s" % fn)
            os.remove(fn)
