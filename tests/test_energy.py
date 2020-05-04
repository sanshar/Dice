#!/usr/bin/python

import os
from math import sqrt
import numpy as N
import struct


def run(args):

    if len(args) != 2 and len(args) != 3:
        raise AssertionError("Wrong number of CLI arguments.")

    trusted_energy_file = "trusted_hci.e"
    if len(args) == 3:
        trusted_energy_file = args[2]

    file1 = open("shci.e", "rb")
    file2 = open(trusted_energy_file, "rb")

    tol = float(args[1])

    calc_e = struct.unpack("d", file1.read(8))[0]
    given_e = struct.unpack("d", file2.read(8))[0]
    if abs(given_e - calc_e) > tol:
        print("\t", given_e, "-", calc_e, " > ", tol)
        print("\tFAILED Energy Test....")
    else:
        print("\tPASSED Energy Test....")


if __name__ == "__main__":
    import sys

    run(sys.argv)
