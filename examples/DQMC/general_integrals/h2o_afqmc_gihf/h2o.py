import numpy as np
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp, cc
from pyscf.shciscf import shci
import sys, os
import scipy.linalg as la
import QMCUtils
import h5py

np.set_printoptions(precision=7, linewidth=1000, suppress=True)

# these need to be provided
vmc_root = "/projects/anma2640/VMC/master/VMC/"
dice_binary = "/projects/anma2640/relDice/Dice/ZDice2"

mol = gto.M(
atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161
''',
basis = 'sto-6g', charge = 1, spin = 1, symmetry = 0, verbose = 3)
norb = mol.nao

mf = scf.RHF(mol)
mf.kernel()

# fci
cisolver = fci.FCI(mf)
e_fci, ci = cisolver.kernel()
print('e(FCI) = %.12f' % e_fci)
dm1_fci = cisolver.make_rdm1(ci, mol.nao, mol.nelec)

# ghf
gmf = scf.GHF(mol)
gmf.kernel()
mo1 = gmf.stability()
dm1 = gmf.make_rdm1(mo1, gmf.mo_occ)
gmf = gmf.run(dm1)
gmf.stability()
dm1_ghf = gmf.make_rdm1()

# dice
# dummy shciscf object for specifying options
mc = shci.SHCISCF(mf, norb, mol.nelectron)
mc.mo_coeff = mf.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-5 ]
mc.fcisolver.DoRDM = False
shci.writeSHCIConfFile(mc.fcisolver, mol.nelec, False)
os.system("mv input.dat dice.dat")
with open("dice.dat", "a") as fh:
  fh.write("readText\n")
  fh.write("writebestdeterminants 10000\n")

QMCUtils.calc_write_hci_ghf_integrals(gmf)

# run dice calculation
print("\nStarting Dice calculation")
command = f'''
              mpirun {dice_binary} dice.dat > dice.out;
              rm -f shci.e; rm -f FCIDUMP;
           '''
os.system(command)
print("Finished Dice calculation\n")

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
QMCUtils.prepAFQMC_gihf(gmf)

# write hf wave function coefficients
norb_act, state, ndets_all = QMCUtils.read_dets_ghf('dets.bin', 1)
occ_first = np.argsort(-np.array(list(state.keys())[0]))
ghfCoeffs = np.eye(2*norb)
QMCUtils.writeMat(ghfCoeffs[:, occ_first.astype(int)], "ghf.txt")

# write afqmc input and perform calculation
afqmc_binary = vmc_root + "/bin/DQMC"

os.system("export OMP_NUM_THREADS=1; rm samples.dat rdm_*.dat -f")

# hci trial
for ndets in [ 1, 10, 100 ]:
  scratchDir = f"rdm_{ndets}"
  os.system(f"mkdir -p {scratchDir}")
  QMCUtils.write_afqmc_input(intType="g", seed=16835, left="multislater", right="ghf", ndets=ndets, nwalk=5, stochasticIter=1000, burnIter=100, choleskyThreshold=1.e-3, writeOneRDM=True, scratchDir=scratchDir, fname=f"afqmc_{ndets}.json")
  print(f"\nStarting AFQMC / HCI ({ndets}) calculation", flush=True)
  command = f'''
                mpirun {afqmc_binary} afqmc_{ndets}.json;
                mv samples.dat samples_{ndets}.dat;
                mv blocking.tmp blocking_{ndets}.out;
                mv afqmc.dat afqmc_{ndets}.dat;
                cat blocking_{ndets}.out;
             '''
  os.system(command)
  norb_dice, state_dice, _ = QMCUtils.read_dets_ghf(ndets=ndets)
  rdm_dice = QMCUtils.calculate_ci_1rdm_ghf(norb_dice, state_dice, ndets=ndets)
  print('\n1e energy')
  obsVar = np.array( [np.einsum('ij,ji->', h1e_g, rdm_dice)] )
  obsMean, obsError = QMCUtils.calculate_observables([ h1e_g ], prefix=scratchDir)
  print(f'{ndets} dets variational e1: {obsVar}')
  print(f'{ndets} dets mixed afqmc e1: {obsMean}')
  print(f'{ndets} dets mixed afqmc e1 errors: {obsError}')
  print(f'{ndets} dets extrapolated e1: {2*obsMean - obsVar}')
  print(f'{ndets} dets extrapolated e1 errors: {2*obsError}')

