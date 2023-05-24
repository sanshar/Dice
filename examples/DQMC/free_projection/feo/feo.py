import numpy
from pyscf import gto, scf, dft, ao2mo, mcscf, tools, lib, symm, cc
from pyscf.tools import molden
import scipy.linalg as la
from pauxy.utils.from_pyscf import generate_integrals
from pyscf.shciscf import shci
import prepVMC

# Checkpoint File Name
r = 1.616
atomString = f'Fe 0 0 0; O {r} 0 0;'
mol = gto.M(atom = atomString, basis = {'Fe': 'ano@6s5p3d2f1g', 'O': 'ano@4s3p2d1f'}, spin = 4, symmetry = 'c2v', verbose = 4)
mf = scf.ROHF(mol).x2c()
mf.level_shift = 0.4
mf.max_cycle = 200
mf.irrep_nelec = { "A1": (10, 9), 'B1': (4, 3), 'B2': (4, 3), 'A2': (1, 0) }
mf.chkfile = 'feo.chk'
#chkfile = 'feo.chk'
#mf.__dict__.update(lib.chkfile.load(chkfile, "scf"))
mf.kernel()
mf.analyze()
#tools.molden.from_mo(mol, 'feo.molden', mf.mo_coeff)
print(f'mol.nao: {mol.nao}')
print(f'mol.nelec: {mol.nelec}')

norbFrozen = 10
ncore = norbFrozen
norbAct = 100
nelecAct = mol.nelectron - 2*norbFrozen

norbAct = mol.nao - norbFrozen
mc = mcscf.CASSCF(mf, norbAct, nelecAct)
mc.mo_coeff = mf.mo_coeff
moFrozen = mc.mo_coeff[:,:norbFrozen]
moActive = mc.mo_coeff[:,norbFrozen:norbFrozen+norbAct]
core_dm = 2 * moFrozen.dot(moFrozen.T)
corevhf = mc.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
moActDice = moActive
h1eff = moActDice.T.dot(mc.get_hcore() + corevhf).dot(moActDice)
eri = ao2mo.kernel(mol, moActDice)
tools.fcidump.from_integrals('FCIDUMP_can', h1eff, eri, norbAct, mol.nelectron - 2*ncore, energy_core)

coeffs = numpy.zeros((norbAct, 2*norbAct))
coeffs[:,:norbAct] = numpy.eye(norbAct)
coeffs[:,norbAct:] = numpy.eye(norbAct)
prepVMC.writeMat(coeffs, 'uhf.txt')

coeffsr = numpy.eye(norbAct)
prepVMC.writeMat(coeffsr, 'rhf.txt')

h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mc.mo_coeff, chol_cut=1e-5, verbose=True)

nbasis = h1e.shape[-1]
rotCorevhf = moActive.T.dot(corevhf).dot(moActive)
h1e = h1e[norbFrozen:norbFrozen+norbAct, norbFrozen:norbFrozen+norbAct] + rotCorevhf
chol = chol.reshape((-1, nbasis, nbasis))
chol = chol[:, norbFrozen:norbFrozen+norbAct, norbFrozen:norbFrozen+norbAct]
mol.nelec = (mol.nelec[0]-norbFrozen, mol.nelec[1]-norbFrozen)
enuc = energy_core

# after core averaging
nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0

chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

myucc = cc.UCCSD(mf)
myucc.frozen = norbFrozen
myucc.verbose = 5
myucc.kernel()
#overlap = mf.get_ovlp(mol)
#rotation = (mc1.mo_coeff[:, norbFrozen:].T).dot(overlap.dot(mf.mo_coeff[:, norbFrozen:]))
prepVMC.write_uccsd(myucc.t1, myucc.t2)

et = myucc.ccsd_t()
print('UCCSD(T) energy', myucc.e_tot + et)
