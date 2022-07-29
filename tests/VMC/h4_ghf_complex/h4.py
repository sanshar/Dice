import numpy as np
from pyscf import gto, scf, ao2mo, tools, fci
from pyscf.lo import pipek
from scipy.linalg import fractional_matrix_power

nsites = 4
norb = 4
sqrt2 = 2**0.5

mol = gto.M(atom = 'H 0 0 0; H 1 0 0; H 2 0 0; H 3 0 0', basis = 'sto-3g')
mf = scf.RHF(mol)
print mf.kernel()
lmo = fractional_matrix_power(mf.get_ovlp(mol), -0.5).T
#lmo = pipek.PM(mol).kernel(mf.mo_coeff)
h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
eri = ao2mo.kernel(mol, lmo)
tools.fcidump.from_integrals('FCIDUMP', h1, eri, 4, 4, mf.energy_nuc())
print mf.mo_coeff
print 'local'
print lmo

cisolver = fci.direct_spin1.FCI(mol)
e, ci = cisolver.kernel(h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc())
print e
print ci.T.flatten()

ovlp = mf.get_ovlp(mol)
gmf = scf.GHF(mol)
gmf.max_cycle = 200
dm = gmf.get_init_guess()
print dm.shape
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
