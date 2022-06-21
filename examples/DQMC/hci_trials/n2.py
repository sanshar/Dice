# N2 ground state

import os
from pyscf import gto, scf, mcscf
from pyscf.shciscf import shci
import QMCUtils

# these need to be provided
dice_binary = "/projects/anma2640/newDice/Dice/Dice"
vmc_root = "/projects/anma2640/VMC/master/VMC"

# build your molecule
r = 2.0
atomstring = f'N 0 0 {-r/2}; N 0 0 {r/2}'
mol = gto.M(atom=atomstring, unit='bohr', basis='ccpvdz', verbose=3, symmetry=1, spin=0)
mf = scf.RHF(mol)
mf.kernel()

norb_frozen = 2

# casscf
norb_act = 8
nelec_act = 10
mc0 = mcscf.CASSCF(mf, norb_act, nelec_act)
mc0.frozen = norb_frozen
mc0.mc1step()

# running dice to write hci wave function
mc = shci.SHCISCF(mf, norb_act, nelec_act)
mc.mo_coeff = mc0.mo_coeff
mc.fcisolver.sweep_iter = [ 0 ]
mc.fcisolver.sweep_epsilon = [ 1e-4 ]
shci.dryrun(mc, mc.mo_coeff)
with open("input.dat", "a") as fh:
  fh.write("writebestdeterminants 10000")
command = f"mv input.dat dice.dat; mpirun {dice_binary} dice.dat > dice.out; rm -f shci.e"
os.system(command)

QMCUtils.run_afqmc(mc, vmc_root = vmc_root, ndets = 100, norb_frozen = norb_frozen)
