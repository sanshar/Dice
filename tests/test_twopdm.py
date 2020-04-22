#!/usr/bin/python

import numpy as np
from rdm_utilities import read_Dice2RDM


def test2RDM(file1: str, file2: str, tol: float):
    """Compare the L2 norm for two RDM files. If they are .txt files, we
    assume they are from Dice and in the Dice RDM format. If they are in .npy
    we assume they're from PySCF and we process them accordingly, i.e. transpose
    some indices.

    Parameters
    ----------
    file1 : str
        RDM file, either from Dice (.txt) or PySCF (.npy).
    file2 : str
        RDM file, either from Dice (.txt) or PySCF (.npy).
    tol : float
        Tolerance for L2 error.
    """

    rdm_1, rdm_2 = None, None

    if file1.endswith("npy"):
        rdm_1 = np.load(file1)
        rdm_1 = rdm_1.transpose(0, 2, 1, 3)
    else:
        rdm_1 = read_Dice2RDM(file1)

    if file2.endswith("npy"):
        rdm_2 = np.load(file2)
        rdm_2 = rdm_2.transpose(0, 2, 1, 3)
    else:
        rdm_2 = read_Dice2RDM(file2)

    l2_norm = np.linalg.norm(rdm_1 - rdm_2)
    if l2_norm > float(tol):
        print("\tFAILED 2RDM Test: L2-Norm of error = {:.3e} ....".format(l2_norm))
    else:
        print("\tPASSED 2RDM Test: L2-Norm of error = {:.3e} ....".format(l2_norm))


if __name__ == "__main__":
    import sys

    test2RDM(sys.argv[1], sys.argv[2], float(sys.argv[3]))
