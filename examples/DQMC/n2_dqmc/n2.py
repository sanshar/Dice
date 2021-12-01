import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
from pauxy.utils.from_pyscf import generate_integrals
import scipy.linalg as la
import prepVMC

r = 2.
atomstring = f'N {-r/2} 0. 0.; N {r/2} 0. 0.;'
mol = gto.M(
    atom=atomstring,
    basis='6-31g',
    verbose=4,
    unit='bohr',
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)
tools.fcidump.from_integrals('FCIDUMP_can', h1, eri, mol.nao, mol.nelectron, mf.energy_nuc())

# set up dqmc calculation
rhfCoeffs = numpy.eye(norb)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")

h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mf.mo_coeff, chol_cut=1e-5, verbose=True)

nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0
chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, filename='FCIDUMP_chol')

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = 0
mycc.verbose = 5
mycc.kernel()
prepVMC.write_ccsd(mycc.t1, mycc.t2)

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
