import os
import numpy
from pyscf import gto, scf, cc, fci
import QMCUtils
import numpy as np
from functools import partial

print = partial(print, flush=True)

r = 1.07
theta = 100.08 * numpy.pi / 180.
rz = r * numpy.sqrt(numpy.cos(theta/2)**2 - numpy.sin(theta/2)**2/3)
dc = 2 * r * numpy.sin(theta/2) / numpy.sqrt(3)
atomstring = f'''
N 0. 0. 0.
H 0. {dc} {rz}
H {r * numpy.sin(theta/2)} {-dc/2} {rz}
H {-r * numpy.sin(theta/2)} {-dc/2} {rz}
'''

mol = gto.M(atom = atomstring, basis = '6-31g', verbose = 3)
mf = scf.RHF(mol)
mf.kernel()

# dipole integrals
nuc_dipmom = [0.0, 0.0, 0.0]
for i in range(mol.natm):
  for j in range(3):
    nuc_dipmom[j] += mol.atom_charge(i) * mol.atom_coord(i)[j]

dip_ints_ao = -mol.intor_symmetric('int1e_r', comp=3)
dip_ints_mo = np.empty_like(dip_ints_ao)
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo[i] = mf.mo_coeff.T.dot(dip_ints_ao[i]).dot(mf.mo_coeff)

# ccsd
mycc = cc.CCSD(mf)
mycc.kernel()

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
dm1_cc = mycc.make_rdm1()
edip_cc = [0., 0. ,0.]
for i in range(3):
  edip_cc[i] = numpy.trace(numpy.dot(dm1_cc, dip_ints_mo[i]))
print(f'orbital unrelaxed cc dipole: {numpy.array(nuc_dipmom) + numpy.array(edip_cc)}')

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

print('\nOrbital relaxed cc dipoles:')
mf = scf.RHF(mol)
dE = 1.e-5
E = numpy.array([ 0., 0., -dE ])
h1e = mf.get_hcore()
h1e += E[2] * dip_ints_ao[2]
mf.get_hcore = lambda *args: h1e
mf.verbose = 1
mf.kernel()
emf_m = mf.e_tot + E[2] * nuc_dipmom[2]

mycc = cc.CCSD(mf)
mycc.kernel()
eccsd_m = mycc.e_tot + E[2] * nuc_dipmom[2]
et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
eccsdpt_m = mycc.e_tot + et + E[2] * nuc_dipmom[2]

E = numpy.array([ 0., 0., dE ])
mf = scf.RHF(mol)
h1e = mf.get_hcore()
h1e += E[2] * dip_ints_ao[2]
mf.get_hcore = lambda *args: h1e
mf.verbose = 1
mf.kernel()
emf_p = mf.e_tot + E[2] * nuc_dipmom[2]

mycc = cc.CCSD(mf)
mycc.kernel()
eccsd_p = mycc.e_tot + E[2] * nuc_dipmom[2]
et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)
eccsdpt_p = mycc.e_tot + et + E[2] * nuc_dipmom[2]

print(f'emf_m: {emf_m}, emf_p: {emf_p}\ndip_mf: {(emf_p - emf_m) / 2 / dE}')
print(f'eccsd_m: {eccsd_m}, eccsd_p: {eccsd_p}\ndip_ccsd: {(eccsd_p - eccsd_m) / 2 / dE}')
print(f'eccsdpt_m: {eccsdpt_m}, eccsd_p: {eccsdpt_p}\ndip_ccsdpt: {(eccsdpt_p - eccsdpt_m) / 2 / dE}')

