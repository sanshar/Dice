import numpy as np
from pyscf import gto, scf,cc, dft, mcscf
from pyscf.shciscf import shci
from pyscf.tools import molden
import QMCUtils
import os

nproc = 3
dice_binary = "/projects/joku8258/software/Dice/Dice"
vmc_root = "/projects/joku8258/QMC/VMC/"


mol = gto.Mole()
mol.atom = '''
C      0.000000    1.392503    0.000000
C     -1.205943    0.696252    0.000000
C     -1.205943   -0.696252    0.000000
C      0.000000   -1.392503    0.000000
C      1.205943   -0.696252    0.000000
C      1.205943    0.696252    0.000000
H     -2.141717    1.236521    0.000000
H     -2.141717   -1.236521    0.000000
H      0.000000   -2.473041    0.000000
H      2.141717   -1.236521    0.000000
H      2.141717    1.236521    0.000000
H      0.000000    2.473041    0.000000
'''

mol.basis = '6-31+G(d)'
mol.symmetry = True
mol.spin = 0 # This is the difference in \alpha and \beta electrons so a value of 2 indicates a triplet.
mol.charge = 0
mol.verbose = 4
mol.build()
mf = scf.RHF(mol)
mf.conv_tol = 1e-9
mf.kernel()

from pyscf.tools import molden
with open('benzene_gs.molden','w') as f:
    molden.header(mol, f)
    molden.orbital_coeff(mol, f, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
    
norbFrozen = 6		# number of electrons to correlate (1s electrons of 6 Carbon)

print("CASSCF calc")
state_id = 0
mc0                  = mcscf.CASSCF(mf, 6, 6)#.state_specific_(state_id) #.state_average_(weights)
mc0.frozen = norbFrozen
mc0.fcisolver.wfnsym = 'Ag'
mc0.fcisolver.spin = 0
mc0.fix_spin_(ss=0)
mo = mc0.sort_mo_by_irrep({'Ag': 0,'B1g': 0, 'B2g': 1, 'B3g':2,'Au': 1, 'B1u':2, 'B2u':0, 'B3u':0}, {'Ag': 6,'B1g': 3, 'B2g': 0, 'B3g':0,'Au': 0, 'B1u':0, 'B2u':5, 'B3u':4})
mc0.mc1step(mo)

print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
mc = shci.SHCISCF(mf, 6, 6)
mc.mo_coeff = mc0.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-5 ]
mc.fcisolver.davidsonTol = 5.e-5
mc.fcisolver.dE = 1.e-6
mc.fcisolver.maxiter = 6
mc.fcisolver.nPTiter = 0
mc.fcisolver.scratchDirectory = "./" #So that RDM files are present in the folder itself
shci.dryrun(mc, mc.mo_coeff)
command = "mv input.dat dice.dat"
os.system(command)
with open("dice.dat", "a") as fh:
  fh.write("DoRDM\nDoSpinRDM\nprintalldeterminants")

# run dice calculation
print("Starting Dice calculation")
command = f"mpirun -np {nproc} {dice_binary} dice.dat > dice.out; rm -f *.bkp shci.e"
os.system(command)
print("Finished Dice calculation\n")

QMCUtils.run_nevpt2(mc,nelecAct=6,numAct=6,norbFrozen=6,nproc=nproc,numSCSamples=1000,diceoutfile="dice.out",vmc_root=vmc_root)



 
