import numpy as np
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp, cc
from pyscf.shciscf import shci
import sys, os
import scipy.linalg as la
import QMCUtils
import h5py

np.set_printoptions(precision=7, linewidth=1000, suppress=True)

# these need to be provided
nproc = 10
vmc_root = "/projects/anma2640/VMC/master/VMC/"

a = 1.4
atomstring = f'''
H 0. 0. {a/(3**0.5)}
H {a/2} 0. {-a/2/(3**0.5)}
H {-a/2} 0. {-a/2/(3**0.5)}
'''

mol = gto.M(atom = atomstring, basis = 'sto-6g', verbose = 3, unit = 'angstrom', symmetry = 0, spin = 1)
norb = mol.nao

mf = scf.RHF(mol)
mf.kernel()

# fci
cisolver = fci.FCI(mf)
e_fci, ci = cisolver.kernel()
print('e(FCI) = %.12f' % e_fci)
dm1_fci = cisolver.make_rdm1(ci, mol.nao, mol.nelec)

# ghf
lmo = QMCUtils.localizeAllElectron(mf, method="lowdin")
gmf = scf.GHF(mol)
staggered_mos = np.zeros((6,3))
staggered_mos[:3,0] = lmo[:,0]
staggered_mos[:3,1] = 0.5 * lmo[:,1]
staggered_mos[3:,1] = 3**0.5 / 2. * lmo[:,1]
staggered_mos[:3,2] = 0.5 * lmo[:,2]
staggered_mos[3:,2] = -3**0.5 / 2. * lmo[:,2]
dm = staggered_mos.dot(staggered_mos.T)
gmf.kernel(dm)
dm1_ghf = gmf.make_rdm1()

# observable integrals
# one-body energy
h1e = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)
h1e_g = gmf.mo_coeff.T.dot(gmf.get_hcore()).dot(gmf.mo_coeff)

e1_ghf = np.trace(np.dot(dm1_ghf, gmf.get_hcore()))
e1_fci = np.trace(np.dot(dm1_fci, h1e))

print(f'e1_ghf: {e1_ghf}')
print(f'e1_fci: {e1_fci}\n')

# afqmc

print("Preparing AFQMC calculation")
# calculate and write integrals
QMCUtils.prepAFQMC_gihf(mol, gmf)

# write hf wave function coefficients
ghf_coeffs = np.eye(2*norb)
QMCUtils.writeMat(ghf_coeffs, "ghf.txt")

# write afqmc input and perform calculation
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

os.system("export OMP_NUM_THREADS=1; rm samples.dat rdm_*.dat -f")

# ghf trial
scratchDir = "rdm_ghf"
os.system(f"mkdir -p {scratchDir}")
burnIter = 100
QMCUtils.write_afqmc_input(intType="g", seed=16835, left="ghf", right="ghf", nwalk=25, stochasticIter=1000, burnIter=burnIter, choleskyThreshold=2.e-3, writeOneRDM=True, scratchDir=scratchDir, fname=f"afqmc_ghf.json")
print(f"\nStarting AFQMC / GHF calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_ghf.json > afqmc_ghf.out;
              mv samples.dat samples_ghf.dat
              python {blocking_script} samples_ghf.dat {burnIter} > blocking_ghf.out;
              cat blocking_ghf.out;
           '''
os.system(command)
print('\n1e energy')
obsVar = e1_ghf
obsMean, obsError = QMCUtils.calculate_observables([ h1e_g ], prefix=scratchDir)
print(f'variational e1: {obsVar}')
print(f'mixed afqmc e1: {obsMean}')
print(f'mixed afqmc e1 errors: {obsError}')
print(f'extrapolated e1: {2*obsMean - obsVar}')
print(f'extrapolated e1 errors: {2*obsError}')
