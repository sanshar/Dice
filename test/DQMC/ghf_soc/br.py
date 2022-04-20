#!/usr/bin/env python
import os,sys,numpy
from pyscf import gto, scf, lib, tools, ao2mo, mcscf
import h5py
import QMCUtils
import numpy as np
import scipy.linalg as la

mol = gto.M(atom = 'Br 0 0 0', basis = 'crenbl', ecp = 'crenbl', symmetry=1, spin = 1)
mf = scf.ROHF(mol)
mf.irrep_nelec = {'A1g':(2,2), 'A1u':(1,0), 'E1ux':(1,1), 'E1uy':(1,1), 'E2gx':(1,1), 'E2gy':(1,1), 'E1gx':(1,1), 'E1gy':(1,1)}
mf.level_shift = 0.1
mf.kernel()
mo_coeff = mf.mo_coeff

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
soc_e, soc_c = numpy.linalg.eigh(soc_ham)

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

# write ghf coeffs
overlap = gmf.get_ovlp(mol)
basis = la.block_diag(mo_coeff, mo_coeff)
ghfCoeffs = basis.T.dot(overlap).dot(gmf.mo_coeff)
QMCUtils.writeMat(ghfCoeffs, "ghf.txt", True)

QMCUtils.prepAFQMC_soc(mol, mf, ecpso.reshape(2*norb, 2*norb))
QMCUtils.write_afqmc_input(seed = 4321, soc=True, left="ghf", right="ghf", nwalk=10, stochasticIter=50, choleskyThreshold=1.e-3, fname="afqmc.json")
