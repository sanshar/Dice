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

    error_per_element = l2_norm / spatialRDM.size

    if error_per_element > float(tol):
        msg = "\tFailed Spin 1RDM Test: Error per Element: {:.3e}\n".format(
            error_per_element
        )
        msg += "\t                   L2-Norm of Error: {:.3e}\n".format(l2_norm)
        msg += "\t                   L\u221E-Norm of Error: {:.3e}\n".format(
            np.max(np.abs(spatialRDM - test_spatial))
        )
        print(msg)
    else:
        msg = "\tPASSED Spin 1RDM Test: Error per Element: {:.3e}".format(
            error_per_element
        )
        print(msg)


if __name__ == "__main__":
    import sys

    test_spin_1RDM(sys.argv[1], sys.argv[2], float(sys.argv[3]))
