import sys, os
import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
from pyscf.shciscf import shci
import QMCUtils

# these need to be provided
nproc = 8
dice_binary = "/projects/anma2640/newDice/Dice/Dice"
vmc_root = "/projects/anma2640/VMC/master/VMC"

# build your molecule
r = 2.0
atomstring = f"N 0 0 {-r/2}; N 0 0 {r/2}"
mol = gto.M(atom = atomstring, basis='6-31g', unit='bohr', symmetry=1)
mf = scf.RHF(mol)
mf.kernel()

# casscf
mc0 = mcscf.CASSCF(mf, 8, 10)
mc0.mc1step()

# dice

# writing input and integrals
print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
mc = shci.SHCISCF(mf, 8, 10)
mc.mo_coeff = mc0.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-5 ]
mc.fcisolver.davidsonTol = 5.e-5
mc.fcisolver.dE = 1.e-6
mc.fcisolver.maxiter = 6
mc.fcisolver.nPTiter = 0
mc.fcisolver.DoRDM = False
shci.dryrun(mc, mc.mo_coeff)
command = "mv input.dat dice.dat"
os.system(command)
with open("dice.dat", "a") as fh:
  fh.write("writebestdeterminants 1000")

# run dice calculation
print("Starting Dice calculation")
command = f"mpirun -np {nproc} {dice_binary} dice.dat > dice.out; rm -f shci.e"
os.system(command)
print("Finished Dice calculation\n")

# afqmc
print("Preparing AFQMC calculation")
rhfCoeffs = numpy.eye(mol.nao)
QMCUtils.writeMat(rhfCoeffs, "rhf.txt")

# dummy mcsscf for core averaging
mc = mcscf.CASSCF(mf, mol.nao, mol.nelectron)
mc.mo_coeff = mc0.mo_coeff
QMCUtils.prepAFQMC(mol, mf, mc)
QMCUtils.write_afqmc_input(seed = 4321, numAct=8, numCore=2, left="multislater", right="rhf", ndets=500, nwalk=10, stochasticIter=50, choleskyThreshold=1.e-3, fname="afqmc.json")
