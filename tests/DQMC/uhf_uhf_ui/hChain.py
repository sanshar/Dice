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
vmc_root = "/projects/anma2640/VMC/dqmc_uihf/VMC/"

r = 2.5
atomstring = ""
for i in range(6):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(atom = atomstring, basis = 'sto-6g', verbose = 3, unit = 'bohr', symmetry = 0, spin = 0)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao

# uhf
dm = [np.zeros((norb, norb)), np.zeros((norb, norb))]
for i in range(norb//2):
  dm[0][2*i, 2*i] = 1.
  dm[1][2*i+1, 2*i+1] = 1.
umf = scf.UHF(mol)
umf.kernel(dm)

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
  fh.write("DoSpinRDM\n")

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

# afqmc

print("Preparing AFQMC calculation")
# calculate and write integrals
QMCUtils.calculate_write_afqmc_uihf_integrals(ham_ints, norb, mol.nelectron, ms=mol.spin, chol_cut=1.e-6)

# write hf wave function coefficients
uhfCoeffs = np.empty((norb, 2*norb))
uhfCoeffs[::,:norb] = np.eye(norb)
uhfCoeffs[::,norb:] = np.eye(norb)
QMCUtils.writeMat(uhfCoeffs, "uhf.txt")

QMCUtils.write_afqmc_input(intType="u", seed=16835, left="uhf", right="uhf", nwalk=20, stochasticIter=50, choleskyThreshold=1.e-3, fname=f"afqmc.json")
