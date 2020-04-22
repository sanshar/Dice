#!/usr/bin/python

import numpy as np
from rdm_utilities import read_Dice1RDM, read_Dice_spin_1RDM


def test_spin_1RDM(spinRDM_file: str, spatialRDM_file: str, tol: float):
    """Trace spin1RDM and compare to spatial 1RDM.

    Parameters
    ----------
    spinRDM_file : str
        RDM file from Dice (.txt).
    spatialRDM_file : str
        RDM file from Dice (.txt).
    tol : float
        Tolerance for L2 error.
    """

    spinRDM = read_Dice_spin_1RDM(spinRDM_file)
    spatialRDM = read_Dice1RDM(spatialRDM_file)

    # Trace over spin
    test_spatial = np.zeros_like(spatialRDM)
    for i in range(spatialRDM.shape[0]):
        for j in range(spatialRDM.shape[0]):
            for k in [0, 1]:
                test_spatial[i, j] += spinRDM[2 * i + k, 2 * j + k]

    l2_norm = np.linalg.norm(spatialRDM - test_spatial)
    if l2_norm > tol:
        print("\tFAILED Spin 1RDM Test: L2-Norm = {:.3e} ....".format(l2_norm))
    else:
        print("\tPASSED Spin 1RDM Test: L2-Norm = {:.3e} ....".format(l2_norm))


if __name__ == "__main__":
    import sys

    test_spin_1RDM(sys.argv[1], sys.argv[2], float(sys.argv[3]))
