import numpy as np


#
# Reading Dice RDMs
#

### Dice 4RDM ###
def read_Dice4RDM(diceFileName, norbs):
    dice = np.zeros((norbs,) * 8)
    with open(diceFileName) as f:
        content = f.readlines()

    for i in range(1, len(content)):
        C0, C1, C2, C3, D3, D2, D1, D0, val = content[i].split()
        dice[
            int(C0), int(C1), int(C2), int(C3), int(D0), int(D1), int(D2), int(D3)
        ] = float(val)

    return dice


### Dice 3RDM ###
def read_Dice3RDM(diceFileName):
    with open(diceFileName) as f:
        content = f.readlines()

    norbs = int(content[0].split()[0])
    dice = np.zeros((norbs,) * 6)

    for i in range(1, len(content)):
        C0, C1, C2, D2, D1, D0, val = content[i].split()
        dice[int(C0), int(C1), int(C2), int(D0), int(D1), int(D2)] = float(val)

    return dice


### Dice 2RDM ###
def read_Dice2RDM(dice2RDMName):
    with open(dice2RDMName) as f:
        content = f.readlines()

    norbs = int(content[0].split()[0])
    dice2RDM = np.zeros((norbs,) * 4)

    for i in range(1, len(content)):
        c0, c1, d1, d0, val = content[i].split()
        dice2RDM[int(c0), int(c1), int(d1), int(d0)] = float(val)

    return dice2RDM


def read_Dice1RDM(filename):
    with open(filename) as f:
        content = f.readlines()

    norbs = int(content[0].split()[0])
    rdm = np.zeros((norbs,) * 2)

    for i in range(1, len(content)):
        c0, d0, val = content[i].split()
        rdm[int(c0), int(d0)] = float(val)

    return rdm


def read_Dice_spin_1RDM(filename):

    with open(filename) as f:
        content = f.readlines()

    norbs = int(content[0].split()[0])
    rdm = np.zeros((norbs, norbs))

    for i in range(1, len(content)):
        c0, d0, val = content[i].split()
        rdm[int(c0), int(d0)] = float(val)

    return rdm


#
# Comparing RDMs
#


def compare_RDM(rdm_1, rdm_2):
    l2_diff = np.linalg.norm(rdm_1 - rdm_2)
    linf_diff = np.max(np.abs(rdm_1 - rdm_2))
    return l2_diff, linf_diff


# # Check nelec
# nelec = np.einsum("ii", fci_dm1)
# print(f"FCI  1RDM Nelec: {nelec}")
# nelec = np.einsum("ii", dice1)
# print(f"SHCI 1RDM Nelec: {nelec}")

# dm1 = np.einsum("ijkj", fci_dm2) / (nelecas - 1)
# nelec = np.einsum("ii", dm1)
# print(f"FCI  2RDM Nelec: {nelec}")

# dm1 = np.einsum("ijkj", dice2) / (nelecas - 1)
# nelec = np.einsum("ii", dm1)
# print(f"SHCI 2RDM Nelec: {nelec}")

# dm2 = np.einsum("ijklmk", fci_dm3) / (nelecas - 2)
# dm1 = np.einsum("ijkj", dm2) / (nelecas - 1)
# nelec = np.einsum("ii", dm1)
# print(f"FCI  3RDM Nelec: {nelec}")

# dm2 = np.einsum("ijklmk", dice3) / (nelecas - 2)
# dm1 = np.einsum("ijkj", dm2) / (nelecas - 1)
# nelec = np.einsum("ii", dm1)
# print(f"SHCI 3RDM Nelec: {nelec}")
# dice_dm2_from_dm3 = dm2
