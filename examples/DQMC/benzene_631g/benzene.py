import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
from pauxy.utils.from_pyscf import generate_integrals
import prepVMC

atomstring='''
C 0.000000 1.396792    0.000000
C 0.000000 -1.396792    0.000000
C 1.209657 0.698396    0.000000
C -1.209657 -0.698396    0.000000
C -1.209657 0.698396    0.000000
C 1.209657 -0.698396    0.000000
H 0.000000 2.484212    0.000000
H 2.151390 1.242106    0.000000
H -2.151390 -1.242106    0.000000
H -2.151390 1.242106    0.000000
H 2.151390 -1.242106    0.000000
H 0.000000 -2.484212    0.000000
'''
mol = gto.M(
    atom=atomstring,
    basis='6-31g',
    verbose=4,
    unit='angstrom',
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()

# calculate core contribution
norbFrozen = 6
norbAct = mol.nao - norbFrozen

nelecAct = mol.nelectron - 2*norbFrozen
mc = mcscf.CASSCF(mf, norbAct, nelecAct)
mc.mo_coeff = mf.mo_coeff
moFrozen = mc.mo_coeff[:,:norbFrozen]
moActive = mc.mo_coeff[:,norbFrozen:]
core_dm = 2 * moFrozen.dot(moFrozen.T)
corevhf = mc.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5

# for dice calculation
h1eff = moActive.T.dot(mc.get_hcore() + corevhf).dot(moActive)
eri = ao2mo.kernel(mol, moActive)
tools.fcidump.from_integrals('FCIDUMP_can', h1eff, eri, norbAct, nelecAct, energy_core)

# set up dqmc calculation
rhfCoeffs = numpy.eye(norbAct)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mf.mo_coeff, chol_cut=1e-5, verbose=True)

nbasis = h1e.shape[-1]
rotCorevhf = moActive.T.dot(corevhf).dot(moActive)
h1e = h1e[norbFrozen:, norbFrozen:] + rotCorevhf
chol = chol.reshape((-1, nbasis, nbasis))
chol = chol[:, norbFrozen:, norbFrozen:]
mol.nelec = (mol.nelec[0]-norbFrozen, mol.nelec[1]-norbFrozen)
enuc = energy_core

# after core averaging
nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
print(chol[0])
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0

chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, filename='FCIDUMP_chol')

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = 6
mycc.verbose = 5
mycc.kernel()

et = mycc.ccsd_t()
print('CCSD(T) correlation energy', mycc.e_corr + et)
