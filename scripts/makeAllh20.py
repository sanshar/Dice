import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
#from pyscf.shciscf import shci, settings
from pyscf.lo import pipek, boys
import sys
from scipy.linalg import fractional_matrix_power

UHF = False
r = float(sys.argv[1])*0.529177
n = 20
norb = 20
order = 2
sqrt2 = 2**0.5

atomstring = ""
for i in range(n):
    atomstring += "H 0 0 %g\n"%(i*r)

mol = gto.M(
    atom = atomstring,
    basis = 'sto-6g',
    verbose=4,
    symmetry=0,
    spin = 0)
mf = scf.RHF(mol)
if (UHF) :
    mf = scf.UHF(mol)
print mf.kernel()

if UHF :
    mocoeff = mf.mo_coeff[0]
else:
    mocoeff = mf.mo_coeff

lowdin = fractional_matrix_power(mf.get_ovlp(mol), -0.5).T
lmo = pipek.PM(mol).kernel(lowdin)
h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
eri = ao2mo.kernel(mol, lmo)
tools.fcidump.from_integrals('FCIDUMP', h1, eri, norb, n, mf.energy_nuc())

#print the atom with which the lmo is associated
orbitalOrder = []
for i in range(lmo.shape[1]):
	orbitalOrder.append(np.argmax(np.absolute(lmo[:,i])) )
print orbitalOrder

ovlp = mf.get_ovlp(mol)
uc = (mf.mo_coeff.T.dot(ovlp).dot(lmo)).T
if UHF:
  uc2 = (mf.mo_coeff[1].T.dot(ovlp).dot(lmo)).T
else:
  uc2 = uc

fileh = open("rhf.txt", 'w')
for i in range(norb):
    for j in range(norb):
        fileh.write('%16.10e '%(uc[i,j]))
    fileh.write('\n')
fileh.close()

fileHF = open("uhf.txt", 'w')
for i in range(norb):
    for j in range(norb):
        fileHF.write('%16.10e '%(uc[i,j]))
    for j in range(norb):
        fileHF.write('%16.10e '%(uc2[i,j]))
    fileHF.write('\n')
fileHF.close()

diag = np.zeros(shape=(norb,norb))
for i in range(norb/2):
  diag[i,i] = 1.
pairMat = uc.dot(diag).dot(uc.T)
filePair = open("pairMatAGP.txt", 'w')
for i in range(norb):
    for j in range(norb):
        filePair.write('%16.10e '%(pairMat[i,j]))
    filePair.write('\n')
filePair.close()

gmf = scf.GHF(mol)
gmf.max_cycle = 200
dm = gmf.get_init_guess()
dm = dm + np.random.rand(2*norb, 2*norb) / 3
print gmf.kernel(dm0 = dm)
uc1 = (gmf.mo_coeff[:norb, :norb].T.dot(ovlp).dot(lmo)).T
uc2 = (gmf.mo_coeff[:norb, norb:].T.dot(ovlp).dot(lmo)).T
uc3 = (gmf.mo_coeff[norb:, :norb].T.dot(ovlp).dot(lmo)).T
uc4 = (gmf.mo_coeff[norb:, norb:].T.dot(ovlp).dot(lmo)).T

fileHF = open("ghf.txt", 'w')
for i in range(norb):
    for j in range(norb):
        fileHF.write('%16.10e '%(uc1[i,j]))
    for j in range(norb):
        fileHF.write('%16.10e '%(uc2[i,j]))
    fileHF.write('\n')
for i in range(norb):
    for j in range(norb):
        fileHF.write('%16.10e '%(uc3[i,j]))
    for j in range(norb):
        fileHF.write('%16.10e '%(uc4[i,j]))
    fileHF.write('\n')
fileHF.close()

ghfCoeffs = np.block([[uc1, uc2], [uc3, uc4]])
theta = ghfCoeffs[::, :n]

amat = np.full((n, n), 0.)
for i in range(n/2):
  amat[2 * i + 1, 2 * i] = -1.
  amat[2 * i, 2 * i + 1] = 1.


pairMat = theta.dot(amat).dot(theta.T)

filePfaff = open("pairMatPfaff.txt", 'w')
for i in range(2*norb):
    for j in range(2*norb):
        filePfaff.write('%16.10e '%(pairMat[i,j]))
    filePfaff.write('\n')
filePfaff.close()

