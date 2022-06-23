#!/usr/bin/python

import numpy as np
from rdm_utilities import (
    read_Dice_spin_1RDM,
    read_Dice1RDM,
    read_Dice2RDM,
    read_Dice3RDM,
)


def test_RDM_nelec(
    FCIDUMP_file: str,
    spatial1RDM_file: str,
    spatial2RDM_file: str,
    spatial3RDM_file: str,
    tol: float,
):

    # Get Nelec from FCIDUMP
    with open(FCIDUMP_file, "r") as f:
        line = f.readline()
        nelec = int(line.split(",")[1].split("=")[1])

    # Get RDMs
    oneRDM = read_Dice1RDM(spatial1RDM_file)
    twoRDM = read_Dice2RDM(spatial2RDM_file)
    threeRDM = read_Dice3RDM(spatial3RDM_file)

    # Check that the shapes match
    norb = oneRDM.shape[0]
    if twoRDM.shape != (norb,) * 4:
        raise ValueError(
            "Shape of spatial2RDM {} doesn't match the shape of spatial1RDM {}".format(
                twoRDM.shape, oneRDM.shape
            )
        )
    if threeRDM.shape != (norb,) * 6:
        raise ValueError(
            "Shape of spatial3RDM {} doesn't match the shape of spatial1RDM {}".format(
                threeRDM.shape, oneRDM.shape
            )
        )

    #
    # Check Nelec
    #
    nelec_from_1RDM = np.einsum("ii", oneRDM)

    dm1_from_2RDM = np.einsum("ijkj", twoRDM) / (nelec - 1)
    nelec_from_2RDM = np.einsum("ii", dm1_from_2RDM)

    dm2_from_3RDM = np.einsum("ijklmk", threeRDM) / (nelec - 2)
    dm1_from_3RDM = np.einsum("ijkj", dm2_from_3RDM) / (nelec - 1)
    nelec_from_3RDM = np.einsum("ii", dm1_from_3RDM)

    # Check 1RDM nelec
    if abs(nelec - nelec_from_1RDM) > tol:
        print(
            "\t\033[91mFailed\033[00m  RDM NELEC from spatial1RDM {:14.10f} != {:d} from FCIDUMP".format(
                nelec_from_1RDM, nelec
            )
        )
    else:
        print("\t\033[92mPassed\033[00m RDM NELEC from spatial1RDM")

    # Check 2RDM nelec
    if abs(nelec - nelec_from_2RDM) > tol:
        print(
            "\t\033[91mFailed\033[00m  RDM NELEC from spatial2RDM {:14.10f} != {:d} from FCIDUMP".format(
                nelec_from_2RDM, nelec
            )
        )
    else:
        print("\t\033[92mPassed\033[00m RDM NELEC from spatial2RDM")

    # Check 3RDM nelec
    if abs(nelec - nelec_from_3RDM) > tol:
        print(
            "\t\033[91mFailed\033[00m  RDM NELEC from spatial3RDM {:14.10f} != {:d} from FCIDUMP".format(
                nelec_from_3RDM, nelec
            )
        )
    else:
        print("\t\033[92mPassed\033[00m RDM NELEC from spatial3RDM")


if __name__ == "__main__":
    import sys

    test_RDM_nelec(
        sys.argv[1],  # FCIDUMP
        sys.argv[2],  # spatial1RDM
        sys.argv[3],  # spatial2RDM
        sys.argv[4],  # spatial3RDM
        float(sys.argv[5]),  # Tol
    )
