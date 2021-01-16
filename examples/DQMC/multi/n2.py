import numpy
import h5py
import scipy.sparse
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools
from pauxy.systems.generic import Generic
from pauxy.utils.from_pyscf import generate_integrals
import scipy.linalg as la
import prepVMC


r = 2.5
atomstring = f'N {-r/2} 0. 0.; N {r/2} 0. 0.;'
mol = gto.M(
    atom=atomstring,
    basis='6-31g',
    verbose=4,
    unit='Bohr',
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
#mf.chkfile = 'scf.chk'
mf.kernel()
#mc = mcscf.CASSCF(mf, M, N)
#mc.chkfile = 'scf.chk'
#mc.kernel()
#e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
#print(ehf, e_tot)
# Rotate by casscf mo coeffs.
#lmo = prepVMC.localizeAllElectron(mf, "lowdin")
lmo = mf.mo_coeff

overlap = mf.get_ovlp(mol)
rhfCoeffs = prepVMC.basisChange(mf.mo_coeff, lmo, overlap)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")

h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)
print(f'eri.shape: {eri.shape}')
tools.fcidump.from_integrals('FCIDUMP_can', h1, eri, mol.nao, mol.nelectron, mf.energy_nuc())

h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), lmo, chol_cut=1e-8, verbose=True)


nbasis = h1e.shape[-1]
print(f'h1 shape: {h1e.shape}')
print(f'nelec: {nelec}')
print(f'nbasis: {nbasis}')
print(f'enuc: {enuc}')
print(f'chol.shape: {chol.shape}')
chol = chol.reshape((-1, nbasis, nbasis))
print(f'chol.shape after: {chol.shape}')
v0 = 0.5 * numpy.einsum('nik,njk->ij', chol, chol, optimize='optimal')
print(f'v0 shape: {v0.shape}')
h1e_mod = h1e - v0

#prepVMC.writeMat(mf.mo_coeff, 'hf.txt')
chol = chol.reshape((chol.shape[0], -1))
print(f'chol.shape: {chol.shape}')
prepVMC.write_dqmc(h1e, h1e_mod, chol, mol.nelectron, mol.nao, enuc, filename='FCIDUMP_chol')

#cisolver = fci.direct_spin1.FCI(mol)
#h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
#eri = ao2mo.kernel(mol, lmo)
#e, ci = cisolver.kernel(h1, eri, h1.shape[1], mol.nelec, ecore=mol.energy_nuc())
#print(f"fci energy: {e}")
