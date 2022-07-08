import numpy as np
from pyscf import gto, scf, ao2mo, tools, fci
from pyscf.lo import pipek
from scipy.linalg import fractional_matrix_power

n = 4
norb = 4
sqrt2 = 2**0.5

ghfCoeffs = np.full((2*norb, 2*norb), 0.)
row = 0
fileh = open("ghf.txt", 'r')
for line in fileh:
  col = 0
  for coeff in line.split():
    ghfCoeffs[row, col]  = float(coeff)
    col = col + 1
  row = row + 1
fileh.close()

theta = ghfCoeffs[::, :n]

amat = np.full((n, n), 0.)
for i in range(n/2):
  amat[2 * i + 1, 2 * i] = -1.
  amat[2 * i, 2 * i + 1] = 1.

#print amat

pairMat = theta.dot(amat).dot(theta.T)
pairMati = np.random.rand(2*norb, 2*norb)/5
pairMati = pairMati - pairMati.T
#randMat = np.random.rand(2*norb,2*norb)
#randMat = (randMat - randMat.T)/10

filePfaff = open("pairMat.txt", 'w')
for i in range(2*norb):
    for j in range(2*norb):
        filePfaff.write('%16.10e '%(pairMat[i,j]))
    filePfaff.write('\n')
filePfaff.close()

filePfaff = open("pairMati.txt", 'w')
for i in range(2*norb):
    for j in range(2*norb):
        filePfaff.write('%16.10e '%(pairMati[i,j]))
    filePfaff.write('\n')
filePfaff.close()
