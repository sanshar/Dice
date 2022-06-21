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
dice_binary = "/projects/anma2640/relDice/Dice/ZDice2"
vmc_root = "/projects/anma2640/VMC/master/VMC/"

mol = gto.M(
atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161
''',
basis = '631g', charge = 1, spin = 1, symmetry = 1, verbose = 3)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao

# fci
cisolver = fci.FCI(mf)
e_fci, ci = cisolver.kernel()
print('e(FCI) = %.12f' % e_fci)
dm1_fci = cisolver.make_rdm1(ci, mol.nao, mol.nelec)

# uhf
umf = scf.UHF(mol)
umf.kernel()
dm1_uhf = umf.make_rdm1()

# dice
print("\nPreparing Dice calculation")
# writing input
# dummy shciscf object for specifying options
mc = shci.SHCISCF(mf, norb, mol.nelectron)
mc.mo_coeff = mf.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-5 ]
mc.fcisolver.davidsonTol = 5.e-5
mc.fcisolver.dE = 1.e-6
mc.fcisolver.maxiter = 6
mc.fcisolver.nPTiter = 0
mc.fcisolver.DoRDM = False
shci.writeSHCIConfFile(mc.fcisolver, mol.nelec, False)
command = "mv input.dat dice.dat"
os.system(command)
with open("dice.dat", "a") as fh:
  fh.write("readText\n")
  fh.write("writebestdeterminants 10000\n")

# constructing and writing ghf integrals from uhf
ham_ints = QMCUtils.calc_uhf_integrals(umf)
QMCUtils.write_hci_ghf_uhf_integrals(ham_ints, norb, mol.nelectron)

# run dice calculation
print("Starting Dice calculation")
command = f'''
              mpirun -np {nproc} {dice_binary} dice.dat > dice.out;
              rm -f shci.e; rm -f FCIDUMP;
           '''
os.system(command)

# get dice energy from output
e1_dice = 0
e_dice = 0
with open('dice.out', 'r') as fh:
  for line in fh:
    if 'one-body' in line:
      ls = line.split()
      e1_dice = float(ls[3])
    if 'E from 2RDM:' in line:
      ls = line.split()
      e_dice = float(ls[3])
print(f'e_dice: {e_dice}')
print("Finished Dice calculation\n")

# observable integrals
# one-body energy
h1e = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)

e1_uhf = np.trace(np.dot(dm1_uhf[0] + dm1_uhf[1], mf.get_hcore()))
e1_fci = np.trace(np.dot(dm1_fci, h1e))

print(f'e1_uhf: {e1_uhf}')
print(f'e1_fci: {e1_fci}\n')

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

dipole_uhf = np.einsum('kij,ji->k', dip_ints_ao, dm1_uhf[0] + dm1_uhf[1]) + np.array(nuc_dipmom)
dipole_fci = np.einsum('kij,ji->k', dip_ints_mo, dm1_fci) + np.array(nuc_dipmom)

print(f'dipole_uhf: {dipole_uhf}')
print(f'dipole_fci: {dipole_fci}\n')

# spin orbitals
dip_ints_mo = [ [ np.zeros((norb, norb)), np.zeros((norb, norb)) ] for i in range(3) ]
for i in range(dip_ints_ao.shape[0]):
  dip_ints_mo[i][0] = umf.mo_coeff[0].T.dot(dip_ints_ao[i]).dot(umf.mo_coeff[0])
  dip_ints_mo[i][1] = umf.mo_coeff[1].T.dot(dip_ints_ao[i]).dot(umf.mo_coeff[1])


# afqmc

print("Preparing AFQMC calculation")
# calculate and write integrals
QMCUtils.calculate_write_afqmc_uihf_integrals(ham_ints, norb, mol.nelectron, ms=mol.spin, chol_cut=1.e-6)

# write hf wave function coefficients
uhfCoeffs = np.empty((norb, 2*norb))
uhfCoeffs[::,:norb] = np.eye(norb)
uhfCoeffs[::,norb:] = np.eye(norb)
QMCUtils.writeMat(uhfCoeffs, "uhf.txt")


# write afqmc input and perform calculation
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

os.system("export OMP_NUM_THREADS=1; rm samples.dat rdm_*.dat -f")


# uhf trial
scratchDir = "rdm_uhf"
os.system(f"mkdir -p {scratchDir}")
burnIter = 100
QMCUtils.write_afqmc_input(intType="u", seed=16835, left="uhf", right="uhf", nwalk=25, stochasticIter=1000, burnIter=burnIter, choleskyThreshold=2.e-3, writeOneRDM=True, scratchDir=scratchDir, fname=f"afqmc_uhf.json")
print(f"\nStarting AFQMC / UHF calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_uhf.json > afqmc_uhf.out;
              mv samples.dat samples_uhf.dat
              python {blocking_script} samples_uhf.dat {burnIter} > blocking_uhf.out;
              cat blocking_uhf.out;
           '''
os.system(command)
print('\n1e energy')
obsVar = e1_uhf
obsMean, obsError = QMCUtils.calculate_observables_uihf([ ham_ints['h1'] ], prefix=scratchDir)
print(f'variational e1: {obsVar}')
print(f'mixed afqmc e1: {obsMean}')
print(f'mixed afqmc e1 errors: {obsError}')
print(f'extrapolated e1: {2*obsMean - obsVar}')
print(f'extrapolated e1 errors: {2*obsError}')

print('\ndipole')
obsVar = dipole_uhf
obsMean, obsError = QMCUtils.calculate_observables_uihf(dip_ints_mo, constants=nuc_dipmom, prefix=scratchDir)
print(f'variational dipole: {obsVar}')
print(f'mixed afqmc dipole: {obsMean}')
print(f'mixed afqmc dipole errors: {obsError}')
print(f'extrapolated dipole: {2*obsMean - obsVar}')
print(f'extrapolated dipole errors: {2*obsError}')
print(f"Finished AFQMC / UHF calculation\n")


# hci trial
for ndets in [ 10, 100 ]:
  scratchDir = f"rdm_{ndets}"
  os.system(f"mkdir -p {scratchDir}")
  QMCUtils.write_afqmc_input(intType="u", seed=16835, left="multislater", right="uhf", nwalk=25, stochasticIter=1000, burnIter=100, choleskyThreshold=2.e-3, writeOneRDM=True, scratchDir=scratchDir, fname=f"afqmc_{ndets}.json")
  print(f"\nStarting AFQMC / HCI ({ndets}) calculation", flush=True)
  command = f'''
                mpirun -np {nproc} {afqmc_binary} afqmc_{ndets}.json > afqmc_{ndets}.out;
                mv samples.dat samples_{ndets}.dat
                python {blocking_script} samples_{ndets}.dat {burnIter} > blocking_{ndets}.out;
                cat blocking_{ndets}.out;
             '''
  os.system(command)
  norb_dice, state_dice = QMCUtils.read_dets(ndets=ndets)
  rdm_dice = QMCUtils.calculate_ci_1rdm(norb_dice, state_dice, ndets=ndets)
  print('\n1e energy')
  obsVar = np.array( [np.einsum('ij,ji->', ham_ints['h1'][0], rdm_dice[0]) + np.einsum('ij,ji->', ham_ints['h1'][1], rdm_dice[1])] )
  obsMean, obsError = QMCUtils.calculate_observables_uihf([ ham_ints['h1'] ], prefix=scratchDir)
  print(f'{ndets} dets variational e1: {obsVar}')
  print(f'{ndets} dets mixed afqmc e1: {obsMean}')
  print(f'{ndets} dets mixed afqmc e1 errors: {obsError}')
  print(f'{ndets} dets extrapolated e1: {2*obsMean - obsVar}')
  print(f'{ndets} dets extrapolated e1 errors: {2*obsError}')

  print('\ndipole')
  obsVar = np.array( [ np.einsum('ij,ji->', dip_ints_mo[i][0], rdm_dice[0]) + np.einsum('ij,ji->', dip_ints_mo[i][1], rdm_dice[1]) for i in range(3) ] )
  obsMean, obsError = QMCUtils.calculate_observables_uihf(dip_ints_mo, constants=nuc_dipmom, prefix=scratchDir)
  print(f'{ndets} dets variational dipole: {obsVar}')
  print(f'{ndets} dets mixed afqmc dipole: {obsMean}')
  print(f'{ndets} dets mixed afqmc dipole errors: {obsError}')
  print(f'{ndets} dets extrapolated dipole: {2*obsMean - obsVar}')
  print(f'{ndets} dets extrapolated dipole errors: {2*obsError}')
  print(f"Finished AFQMC / HCI ({ndets}) calculation\n")

