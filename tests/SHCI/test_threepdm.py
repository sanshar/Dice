#!/usr/bin/python

import numpy as np
from rdm_utilities import read_Dice3RDM


def test3RDM(file1: str, file2: str, tol: float):
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
        rdm_1 = rdm_1.transpose(0, 2, 4, 1, 3, 5)
    else:
        rdm_1 = read_Dice3RDM(file1)

    if file2.endswith("npy"):
        rdm_2 = np.load(file2)
        rdm_2 = rdm_2.transpose(0, 2, 4, 1, 3, 5)
    else:
        rdm_2 = read_Dice3RDM(file2)

    l2_norm = np.linalg.norm(rdm_1 - rdm_2)
    error_per_element = l2_norm / rdm_1.size

    if error_per_element > float(tol):
        msg = "\t\033[91mFailed\033[00m  3RDM Test: Error per Element: {:.3e}\n".format(
            error_per_element
        )
        msg += "\t                   L2-Norm of Error: {:.3e}\n".format(l2_norm)
        msg += "\t                   L\u221E-Norm of Error: {:.3e}\n".format(
            np.max(np.abs(rdm_1 - rdm_2))
        )
        print(msg)
    else:
        msg = "\t\033[92mPassed\033[00m 3RDM Test: Error per Element: {:.3e}".format(
            error_per_element
        )
        print(msg)


if __name__ == "__main__":
    import sys

    test3RDM(sys.argv[1], sys.argv[2], float(sys.argv[3]))
