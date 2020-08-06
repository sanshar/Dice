import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
from pyscf.lo import pipek, boys
import sys
from scipy.linalg import fractional_matrix_power
from scipy.stats import ortho_group

UHF = False
r = float(sys.argv[1])*0.529177
n = 10
order = 2
sqrt2 = 2**0.5

atomstring = ""
for i in range(n):
    atomstring += "H 0 0 %g\n"%(i*r)

mol = gto.M(
    atom = atomstring,
    basis = 'ccpvdz',
    verbose=4,
    symmetry=0,
    spin = 0)
mf = scf.RHF(mol)
print mf.kernel()

mocoeff = mf.mo_coeff

lowdin = fractional_matrix_power(mf.get_ovlp(mol), -0.5).T
pm = pipek.PM(mol).kernel(lowdin)
lmo = np.full((50, 50), 0.)
orth = ortho_group.rvs(dim=5)
for i in range(10):
  lmo[::,5*i:5*(i+1)] = pm[::,5*i:5*(i+1)].dot(orth)

norb = mf.mo_coeff.shape[0]
h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
eri = ao2mo.kernel(mol, lmo)
tools.fcidump.from_integrals('FCIDUMP', h1, eri, norb, n, mf.energy_nuc())

print mf.mo_coeff
print "local"
print lmo

#print the atom with which the lmo is associated
orbitalOrder = []
for i in range(lmo.shape[1]):
	orbitalOrder.append(np.argmax(np.absolute(lmo[:,i])) )
print orbitalOrder

ovlp = mf.get_ovlp(mol)
gmf = scf.GHF(mol)
dm = gmf.get_init_guess()
print dm.shape
dm = dm + np.random.rand(2*norb, 2*norb) / 5
print gmf.kernel(dm0 = dm)


ovlp = mf.get_ovlp(mol)
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
