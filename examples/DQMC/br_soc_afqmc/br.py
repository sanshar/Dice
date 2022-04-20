#!/usr/bin/env python
import os,sys,numpy
from pyscf import gto, scf, lib, tools, ao2mo, mcscf
from pyscf.shciscf import shci, socutils
import h5py
import prepVMC
import numpy as np
import scipy.linalg as la

# write cholesky integrals
def write_dqmc_soc(hcore, hcore_mod, chol, nelec, nmo, enuc, filename='FCIDUMP_chol'):
    assert len(chol.shape) == 2
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, chol.shape[0]])
        fh5['hcore_real'] = hcore.real.flatten()
        fh5['hcore_imag'] = hcore.imag.flatten()
        fh5['hcore_mod_real'] = hcore_mod.real.flatten()
        fh5['hcore_mod_imag'] = hcore_mod.imag.flatten()
        fh5['chol'] = chol.flatten()
        fh5['energy_core'] = enuc

mol = gto.M(
    atom = 'Br 0 0 0',
    basis = 'crenbl',
    ecp = 'crenbl',
    verbose=4,
    symmetry=1,
    spin = 1)

mf = scf.ROHF(mol)
mf.irrep_nelec = {'A1g':(2,2), 'A1u':(1,0), 'E1ux':(1,1), 'E1uy':(1,1), 'E2gx':(1,1), 'E2gy':(1,1), 'E1gx':(1,1), 'E1gy':(1,1)}
mf.level_shift = 0.1
mf.kernel()
mf.analyze()
rohf_dm = mf.make_rdm1()
mo_coeff = mf.mo_coeff

energy_core = mol.energy_nuc()
moAct = mo_coeff
h1e = moAct.T.dot(mf.get_hcore()).dot(moAct)
eri = ao2mo.kernel(mol, moAct)
tools.fcidump.from_integrals('FCIDUMP', h1e, eri, mol.nao, mol.nelectron, energy_core)
mc = mcscf.CASSCF(mf, mol.nao, sum(mol.nelec))
mc.make_rdm1 = lambda *args: rohf_dm
mc.mo_coeff = mo_coeff
socutils.writeSOCIntegrals(mc, pictureChange1e = "bp", pictureChange2e = "bp")

mol.symmetry = 0
gmf = scf.GHF(mol)
#mf.with_soc = True
s = .5 * lib.PauliMatrices
ecpso = -1j * lib.einsum('sxy,spq->xpyq', s, mol.intor('ECPso'))
hcore = gmf.get_hcore()
hcore = hcore + ecpso.reshape(hcore.shape)
gmf.get_hcore = lambda *args: hcore
norb = mol.nao
ncore = (sum(mol.nelec) - 5)//2

# diagonalize soc ham in p space to get initial guess for ghf
porbs = numpy.zeros((2*norb, 6))
for i in range(3):
  porbs[:norb, 2*i] = mo_coeff[:, ncore + i]
  porbs[norb:, 2*i+1] = mo_coeff[:, ncore + i]
soc_ham = porbs.T.dot(ecpso.reshape(hcore.shape)).dot(porbs)
numpy.set_printoptions(4, linewidth=1000, suppress=True, threshold=sys.maxsize)
print(f'\nsoc_ham:\n{soc_ham}\n')
soc_e, soc_c = numpy.linalg.eigh(soc_ham)
print(f'\nenergy: {soc_e}\n')
print(f'\nstates:\n{soc_c}\n')

soc_orbs = porbs.dot(soc_c)
ghf_coeffs_guess = numpy.zeros((2*norb, sum(mol.nelec))) * 1.j
ghf_coeffs_guess[:norb, :ncore] = mo_coeff[:, :ncore]
ghf_coeffs_guess[norb:, ncore:2*ncore] = mo_coeff[:, :ncore]

# ground state
for i in range(5):
  ghf_coeffs_guess[:, 2*ncore+i] = soc_orbs[:, i]
dm = ghf_coeffs_guess.dot(ghf_coeffs_guess.T.conj())
gmf.level_shift = 0.2
gmf.max_cycle = 100
gmf.kernel(dm0=dm)
gmf.analyze()

# excited state
gmf_1 = scf.GHF(mol)
gmf_1.get_hcore = lambda *args: hcore
ghf_coeffs_guess = numpy.zeros((2*norb, sum(mol.nelec))) * 1.j
ghf_coeffs_guess[:norb, :ncore] = mo_coeff[:, :ncore]
ghf_coeffs_guess[norb:, ncore:2*ncore] = mo_coeff[:, :ncore]
for i in range(5):
  ghf_coeffs_guess[:, 2*ncore+i] = soc_orbs[:, i+1]
dm = ghf_coeffs_guess.dot(ghf_coeffs_guess.T.conj())
gmf_1.level_shift = 0.2
gmf_1.max_cycle = 100
gmf_1.kernel(dm0=dm)
gmf_1.analyze()

# write ghf coeffs
overlap = gmf.get_ovlp(mol)
basis = la.block_diag(mo_coeff, mo_coeff)
ghfCoeffs = basis.T.dot(overlap).dot(gmf.mo_coeff)
prepVMC.writeMat(ghfCoeffs, "ghf.txt", True)
ghfCoeffs_1 = basis.T.dot(overlap).dot(gmf_1.mo_coeff)
prepVMC.writeMat(ghfCoeffs_1, "ghf_1.txt", True)

# calculate and write cholesky integrals
h1e, chol, nelec, enuc = prepVMC.generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut=1e-5, verbose=True)

nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')

h1e = basis.T.dot(hcore).dot(basis)
h1e_mod = h1e - numpy.block([ [v0, numpy.zeros((nbasis, nbasis)) ] , [ numpy.zeros((nbasis, nbasis)), v0 ] ])
chol = chol.reshape((chol.shape[0], -1))
write_dqmc_soc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, filename='FCIDUMP_chol')

