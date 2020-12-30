import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools
from pauxy.utils.from_pyscf import generate_integrals
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
mf.kernel()

# for vmc
lmo = prepVMC.localizeAllElectron(mf, "lowdin")
prepVMC.writeFCIDUMP(mol, mf, lmo, 'FCIDUMP_eri')

overlap = mf.get_ovlp(mol)
rhfCoeffs = prepVMC.basisChange(mf.mo_coeff, lmo, overlap)
prepVMC.writeMat(rhfCoeffs, "hf.txt")

# for dice calculation
h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff)
print(f'eri.shape: {eri.shape}')
tools.fcidump.from_integrals('FCIDUMP_can', h1, eri, mol.nao, mol.nelectron, mf.energy_nuc())

# set up dqmc calculation
prepVMC.writeMat(rhfCoeffs, "rhf.txt")
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), lmo, chol_cut=1e-5, verbose=True)

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

