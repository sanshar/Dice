# ethylene 1 1^b2u excited state

import os
import numpy
from pyscf import gto, scf, mcscf
from pyscf.shciscf import shci
import QMCUtils

# these need to be provided
dice_binary = "/projects/anma2640/newDice/Dice/Dice"
vmc_root = "/projects/anma2640/VMC/master/VMC"

# build your molecule
r = 1.085
theta = 102 * numpy.pi / 180.
atomstring = '''
                C 0.00000000 1.26026583 0.00000000
                C 0.00000000 -1.26026583 0.00000000
                H 0.00000000 2.32345976 1.74287672
                H 0.00000000 -2.32345976 1.74287672
                H 0.00000000 2.32345976 -1.74287672
                H 0.00000000 -2.32345976 -1.74287672
             '''
mol = gto.M(atom=atomstring, unit='bohr', basis='6-31g', verbose=3, symmetry=1, spin=0)
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 2

# casscf
norb_act = 2
nelec_act = 2
mc0 = mcscf.CASSCF(mf, norb_act, nelec_act)
mc0.fcisolver.wfnsym = 'B2u'
mo = mc0.sort_mo_by_irrep({'ag': 0, 'b1g': 1, 'b2u': 0, 'b3u': 1}, {'ag': 3, 'b2u': 2, 'b1u': 1, 'b3g': 1})
mc0.frozen = norb_frozen
mc0.mc1step(mo)

# running dice to write hci wave function
mc = shci.SHCISCF(mf, norb_act, nelec_act)
mc.mo_coeff = mc0.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-4 ]
mc.fcisolver.treversal = 1
mc.fcisolver.initialStates = [ [ 0, 3 ] ]
shci.dryrun(mc, mc.mo_coeff)
with open("input.dat", "a") as fh:
  fh.write("writebestdeterminants 10000")
command = f"mv input.dat dice.dat; mpirun {dice_binary} dice.dat > dice_b2u.out; rm -f shci.e"
os.system(command)

QMCUtils.run_afqmc(mc, vmc_root = vmc_root, norb_frozen = norb_frozen)
