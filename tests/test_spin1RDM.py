import sys
import numpy as np
from scipy.sparse import coo_matrix

#
# Setup
#
spin_rdm_file = "spin1RDM.0.0.txt"
spatial_rdm_file = "spatial1RDM.0.0.txt"

tol = 1e-5
if len(sys.argv) == 2:
    tol = float(sys.argv[1])


#
# Read RDMs
#
N_spin_orbs, N_spatial_orbs = (0, 0)
with open(spin_rdm_file, "r") as f:
    line = f.readline()
    N_spin_orbs = int(line.split()[0])
with open(spatial_rdm_file, "r") as f:
    line = f.readline()
    N_spatial_orbs = int(line.split()[0])

# Test matrix sizes
if N_spatial_orbs != int(N_spin_orbs / 2):
    raise AssertionError("SpinRDM and SpatialRDM have uncompatible dimensions")

#
# Load matrices
#
data = np.loadtxt(spin_rdm_file, skiprows=1)
spinRDM = coo_matrix(
    (data[:, 2], (data[:, 0], data[:, 1])), shape=(N_spin_orbs, N_spin_orbs)
).toarray()

data = np.loadtxt(spatial_rdm_file, skiprows=1)
spatialRDM = coo_matrix(
    (data[:, 2], (data[:, 0], data[:, 1])), shape=(N_spatial_orbs, N_spatial_orbs)
).toarray()

#
# Compare Matrices
#
new_spatialRDM = np.zeros_like(spatialRDM)
for i in range(spatialRDM.shape[0]):
    for j in range(spatialRDM.shape[1]):
        for spin1 in [0, 1]:
            for spin2 in [0, 1]:
                new_spatialRDM[i, j] += spinRDM[2 * i + spin1, 2 * j + spin2]

diff = np.linalg.norm(new_spatialRDM - spatialRDM)
if diff > tol:
    raise AssertionError(
        "Difference in RDMs ({}) is above tolerance ({})".format(diff, tol)
    )
else:
    print("\tPassed spin1RDM test...")
