import numpy as np
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp, cc
from pyscf.shciscf import shci
import sys, os
import scipy.linalg as la
import QMCUtils
import h5py

np.set_printoptions(precision=7, linewidth=1000, suppress=True)

# these need to be provided
nproc = 4
vmc_root = "/projects/anma2640/VMC/master/VMC/"
dice_binary = "/projects/anma2640/relDice/Dice/ZDice2"

a = 1.4
atomstring = f'''
H 0. 0. {a/(3**0.5)}
H {a/2} 0. {-a/2/(3**0.5)}
H {-a/2} 0. {-a/2/(3**0.5)}
'''

mol = gto.M(atom = atomstring, basis = 'sto-6g', verbose = 3, unit = 'angstrom', spin = 1)
norb = mol.nao

mf = scf.RHF(mol)
mf.kernel()

# ghf
gmf = scf.GHF(mol)
gmf.kernel()
mo1 = gmf.stability()
dm1 = gmf.make_rdm1(mo1, gmf.mo_occ)
gmf = gmf.run(dm1)
gmf.stability()

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
              mpirun -np {nproc} {dice_binary} dice.dat > dice.out;
              rm -f shci.e; rm -f FCIDUMP;
           '''
os.system(command)
print("Finished Dice calculation\n")

# afqmc

print("Preparing AFQMC calculation")
# calculate and write integrals
QMCUtils.prepAFQMC_gihf(gmf)

# write hf wave function coefficients
ghf_coeffs = np.eye(2*norb)
QMCUtils.writeMat(ghf_coeffs, "ghf.txt")

# write afqmc input and perform calculation
afqmc_binary = vmc_root + "/bin/DQMC"

os.system("export OMP_NUM_THREADS=1; rm samples.dat rdm_*.dat -f")

# hci trial
ndets = 5
QMCUtils.write_afqmc_input(intType="g", seed=16835, left="multislater", right="ghf", ndets=ndets, nwalk=20, stochasticIter=50, burnIter=50, choleskyThreshold=1.e-3, fname=f"afqmc.json")
print(f"\nStarting AFQMC / HCI ({ndets}) calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc.json;
              mv samples.dat samples.ref;
              mv blocking.tmp blocking.out;
              mv afqmc.dat afqmc.ref;
              cat blocking.out;
           '''
os.system(command)
