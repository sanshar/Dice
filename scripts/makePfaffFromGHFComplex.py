import numpy as np
from pyscf import gto, scf, ao2mo, tools, fci
from pyscf.lo import pipek
from scipy.linalg import fractional_matrix_power

#n = nelec
n = 50
norb = 50
sqrt2 = 2**0.5

matr = np.full((2*norb, 2*norb), 0.)
mati = np.full((2*norb, 2*norb), 0.)

row = 0
fileh = open("ghf.txt", 'r')
for line in fileh:
  col = 0
  for coeff in line.split():
    m = coeff.strip()[1:-1]
    matr[row, col], mati[row, col] = [float(x) for x in m.split(',')]
    col = col + 1
  row = row + 1
fileh.close()

ghf = matr + 1j * mati
theta = ghf[::,:n]

amat = np.full((n, n), 0.)
for i in range(n/2):
  amat[2 * i + 1, 2 * i] = -1.
  amat[2 * i, 2 * i + 1] = 1.

pairMat = theta.dot(amat).dot(theta.T)
pairMatr = pairMat.real
pairMati = pairMat.imag
#randMat = np.random.rand(2*norb,2*norb)
#randMat = (randMat - randMat.T)/10

filePfaff = open("pairMat.txt", 'w')
for i in range(2*norb):
    for j in range(2*norb):
        filePfaff.write('%16.10e '%(pairMatr[i,j]))
    filePfaff.write('\n')
filePfaff.close()

filePfaff = open("pairMati.txt", 'w')
for i in range(2*norb):
    for j in range(2*norb):
        filePfaff.write('%16.10e '%(pairMati[i,j]))
    filePfaff.write('\n')
filePfaff.close()
