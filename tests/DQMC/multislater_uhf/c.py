import sys, os
import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
from pyscf.shciscf import shci
import QMCUtils

nproc = 4
dice_binary = '/projects/anma2640/newDice/Dice/Dice'

# build your molecule
mol = gto.M(atom="C 0 0 0", basis='ccpvdz', symmetry='dooh', spin=2)
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1g':(2,2), 'A1u':(0,0), 'E1ux':(1,0), 'E1uy':(1,0)}
mf.kernel()

# casscf
mc0 = mcscf.CASSCF(mf, 8, 4)
mo = mc0.sort_mo_by_irrep({'A1g': 2, 'A1u': 2, 'E1ux': 2, 'E1uy': 2}, {'A1g': 1})
mc0.frozen = 1
mc0.mc1step(mo)

# dice

# writing input and integrals
print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
mc = shci.SHCISCF(mf, 8, 4)
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
  fh.write("writebestdeterminants 10000")

# run dice calculation
print("Starting Dice calculation")
command = f"mpirun -np {nproc} {dice_binary} dice.dat > dice.out; rm -f shci.e"
os.system(command)
print("Finished Dice calculation\n")

# afqmc

print("Preparing AFQMC calculation")
# write hf wave function coefficients
# rohf states are treated as uhf
rhfCoeffs = numpy.eye(mol.nao)
uhfCoeffs = numpy.block([ rhfCoeffs, rhfCoeffs ])
QMCUtils.writeMat(uhfCoeffs, "uhf.txt")

# calculate and write cholesky integrals
# dummy mcsscf for core averaging
mc = mcscf.CASSCF(mf, mol.nao, mol.nelectron)
mc.mo_coeff = mc0.mo_coeff
QMCUtils.prepAFQMC(mol, mf, mc)

QMCUtils.write_afqmc_input(seed=142108, numAct=8, numCore=1, left="multislater", ndets=100, right="uhf", nwalk=20, stochasticIter=50, choleskyThreshold=2.e-3, fname="afqmc.json")
