import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools
from pyscf.shciscf import shci
import os
import QMCUtils 
import json


nproc = 4
dice_binary = "/projects/joku8258/software/Dice/Dice"
vmc_root = "/projects/joku8258/QMC/VMC/"


# make your molecule here
r = 2.5 * 0.529177
atomstring = "N 0 0 0; N 0 0 %g"%(r)
mol = gto.M(
    atom = atomstring,
    basis = 'augccpvdz',
    verbose=4,
    symmetry=0,
    spin = 0)

mf = scf.RHF(mol)
mf.chkfile = 'N2_HF.chk'
mf.kernel()

norbAct = 8
nelecAct = 10
norbFrozen = 0
#norbCore = int((sum(mol.nelec)-(nelecAct))/2)

########### CASSCF calculation ##################
mc = shci.SHCISCF(mf, norbAct, nelecAct)
mc.chkfile = 'N2_SHCISCF.chk'
mc.fcisolver.sweep_iter = [0]
mc.fcisolver.sweep_epsilon = [0]
mc.fcisolver.nPTiter = 0
mc.max_cycle_macro = 30
mc.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc.kernel()

##########################################################

############# Dice Calculation ###########################
print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
mch = shci.SHCISCF(mf,norbAct, nelecAct)
mch.mo_coeff = mc.mo_coeff
mch.fcisolver.sweep_iter = [ 0 ]
mch.fcisolver.sweep_epsilon = [ 0.0 ]
mch.fcisolver.davidsonTol = 5.e-5
mch.fcisolver.dE = 1.e-8
mch.fcisolver.maxIter = 20
mch.fcisolver.nPTiter = 0
mch.fcisolver.targetError= 1e-5

mc.fcisolver.DoRDM = False
mch.fcisolver.scratchDirectory = "./"
shci.dryrun(mch, mch.mo_coeff)
#exit(0)
command = "mv input.dat dice.dat"
os.system(command)
with open("dice.dat", "a") as fh:
	fh.write("DoRDM\nDoSpinRDM\n printalldeterminants")	# These keywords should be included
fh.close()


# run dice calculation
print("Starting Dice calculation")
command = f"mpirun -np {nproc} {dice_binary} dice.dat > dice.out; rm -f *.bkp shci.e"
os.system(command)
print("Finished Dice calculation\n")


################################################################################
QMCUtils.run_nevpt2(mc,nelecAct=nelecAct,numAct=norbAct,norbFrozen=norbFrozen,nproc=nproc,numSCSamples=1000,diceoutfile="dice.out",vmc_root=vmc_root)

