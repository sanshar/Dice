import numpy
from pyscf import gto, scf, dft, ao2mo, mcscf, tools, lib, symm, cc
from pyscf.tools import molden
import scipy.linalg as la
from pauxy.utils.from_pyscf import generate_integrals
from pyscf.shciscf import shci
import prepVMC

mol = gto.Mole(symmetry=False)
mol.atom = '''C 0 0 0'''
mol.verbose = 4
mol.symmetry = False
mol.spin = 2
mol.basis = 'augccpvtz'
mol.build()

mf = scf.ROHF(mol)
mf.kernel()
norb = mol.nao

# casscf
norbAct = 8
nelecAct = 4
norbFrozen = 1
mc = mcscf.CASSCF(mf, norbAct, nelecAct)
mc.frozen = norbFrozen
mc.mc1step()

# core averaged integrals for dice
moFrozen = mc.mo_coeff[:,:norbFrozen]
core_dm = 2 * moFrozen.dot(moFrozen.T)
corevhf = mc.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
moActDice = mc.mo_coeff[:, norbFrozen:norbAct + norbFrozen]
h1eff = moActDice.T.dot(mc.get_hcore() + corevhf).dot(moActDice)
eri = ao2mo.kernel(mol, moActDice)
tools.fcidump.from_integrals('FCIDUMP_can', h1eff, eri, norbAct, nelecAct, energy_core)


# set up dqmc calculation
# this is rohf really
norbAct = mol.nao - norbFrozen
overlap = mf.get_ovlp(mol)
rhfCoeffs = numpy.eye(norbAct)
rhfCoeffs = prepVMC.basisChange(mf.mo_coeff[:, norbFrozen:], mc.mo_coeff[:, norbFrozen:], overlap)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")

# also rohf
uhfCoeffs = numpy.zeros((norbAct, 2*norbAct))
uhfCoeffs[:,:norbAct] = rhfCoeffs
uhfCoeffs[:,norbAct:] = rhfCoeffs
prepVMC.writeMat(uhfCoeffs, "uhf.txt")

# generating choleskies
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mc.mo_coeff, chol_cut=1e-5, verbose=True)

# core averaging choleskies
moAct = mc.mo_coeff[:,norbFrozen:]
nbasis = h1e.shape[-1]
rotCorevhf = moAct.T.dot(corevhf).dot(moAct)
h1e = h1e[norbFrozen:, norbFrozen:] + rotCorevhf
chol = chol.reshape((-1, nbasis, nbasis))
chol = chol[:, norbFrozen:, norbFrozen:]
mol.nelec = (mol.nelec[0]-norbFrozen, mol.nelec[1]-norbFrozen)
enuc = energy_core
nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0
chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = norbFrozen
mycc.verbose = 5
mycc.kernel()
#prepVMC.write_ccsd(mycc.t1, mycc.t2)

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
