import os
import numpy
from pyscf import gto, scf, cc, fci
import QMCUtils
import numpy as np
from functools import partial

print = partial(print, flush=True)

r = 0.9578
theta = 104.5078 * numpy.pi / 180.
atomstring = f'O 0. 0. 0.; H {r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.; H {-r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.'
mol = gto.M(atom = atomstring, basis = '6-31g', verbose = 3, symmetry = 1)
mf = scf.RHF(mol)
mf.kernel()

# dipole moment
nuc_dipmom = [0.0, 0.0, 0.0]
for i in range(mol.natm):
  for j in range(3):
    nuc_dipmom[j] += mol.atom_charge(i) * mol.atom_coord(i)[j]

# spatial orbitals
dip_ints_ao = -mol.intor_symmetric('int1e_r', comp=3)
dip_ints_mo = np.empty_like(dip_ints_ao)
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo[i] = mf.mo_coeff.T.dot(dip_ints_ao[i]).dot(mf.mo_coeff)

cisolver = fci.FCI(mf)
fci_ene, fci_vec = cisolver.kernel()
print(f'fci_ene: {fci_ene}', flush=True)
dm1 = cisolver.make_rdm1(fci_vec, mol.nao, mol.nelec)
np.savetxt('rdm1_fci.txt', dm1)
h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
print(f'1e ene: {np.trace(np.dot(dm1, h1))}')
dipole_fci = np.einsum('kij,ji->k', dip_ints_mo, dm1) + np.array(nuc_dipmom)
print(f'dipole fci: {dipole_fci}', flush=True)

#QMCUtils.run_afqmc(mf, nwalk_per_proc = 20, dt = 0.01, cholesky_threshold = 0., nblocks = 500)

print("Calculating Cholesky integrals")
h1e, chol, nelec, enuc = QMCUtils.generate_integrals(mol, mf.get_hcore(), mf.mo_coeff, 1.e-5)
nbasis = h1e.shape[-1]
nelec = mol.nelec
chol = chol.reshape((-1, nbasis, nbasis))
v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
h1e_mod = h1e - v0
chol = chol.reshape((chol.shape[0], -1))
QMCUtils.write_dqmc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')
print("Finished calculating Cholesky integrals\n", flush=True)

os.system('export OMP_NUM_THREDAS=1; export MKL_NUM_THREADS=1; mpirun python mpi_jax.py')
obsMean, obsError = QMCUtils.calculate_observables([ h1e ])
print(f'1e_ph: {obsMean}')
print(f'1e_ph error: {obsError}')
obsMean, obsError = QMCUtils.calculate_observables(dip_ints_mo, constants=nuc_dipmom)
print(f'dipole_ph: {obsMean}')
print(f'dipole_ph error: {obsError}')


