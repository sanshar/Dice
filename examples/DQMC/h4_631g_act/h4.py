import numpy
import h5py
import scipy.sparse
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools
from pauxy.utils.from_pyscf import generate_integrals
import prepVMC

r = 4.
atomstring = ""
for i in range(4):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(
    atom=atomstring,
    basis='6-31g',
    verbose=4,
    unit='Bohr',
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()

# set up active space vmc
nact = 4
mo = mf.mo_coeff.copy()
lmo = prepVMC.localizeValence(mf, mo[:,:nact], "ibo")
mo[:,:nact] = lmo

overlap = mf.get_ovlp(mol)
rhfCoeffs = prepVMC.basisChange(mf.mo_coeff, mo, overlap)
prepVMC.writeMat(rhfCoeffs[:nact, :nact], "hf.txt")

h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
eri = ao2mo.kernel(mol, lmo)
print(f'eri.shape: {eri.shape}')
tools.fcidump.from_integrals('FCIDUMP_eri', h1, eri, nact, mol.nelectron, mf.energy_nuc())

# set up full space dqmc
prepVMC.writeMat(rhfCoeffs, "rhf.txt")
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo, chol_cut=1e-5, verbose=True)
nbasis = h1e.shape[-1]
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'enuc: {enuc}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0

chol = chol.reshape((chol.shape[0], -1))
prepVMC.write_dqmc(h1e, h1e_mod, chol, mol.nelectron, mol.nao, enuc)

# fci
cisolver = fci.direct_spin1.FCI(mol)
h1 = mo.T.dot(mf.get_hcore()).dot(mo)
eri = ao2mo.kernel(mol, mo)
e, ci = cisolver.kernel(h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc())
print(f"fci energy: {e}")
