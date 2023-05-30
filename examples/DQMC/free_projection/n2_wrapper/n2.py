import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
import fp_helper
import QMCUtils
from pyscf.shciscf import shci
import os
r = 2.
atomstring = f'N {-r/2} 0. 0.; N {r/2} 0. 0.;'
mol = gto.M(
    atom=atomstring,
    basis='6-31g',
    verbose=4,
    unit='bohr',
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao
norbFrozen = 0

e_guess = -109.0914

vmc_root =  '/projects/joku8258/software/alpine_software/Dice/'
dice_binary = '/projects/xuwa0145/tools/Dice/bin_gcc_alpine/Dice'

##############  RHF/RHF FP-AFQMC ###############
fp_helper.run_fpafqmc(mf,norb_frozen=norbFrozen,vmc_root=vmc_root,ene0Guess= e_guess, chol_cut=1e-5,eneSteps = [ 30, 40, 50, 60 ],dt = 0.05,errorTargets = [0.7e-3,0.7e-3,0.7e-3,0.7e-3],choleskyThreshold = 1e-3,nsteps =60)
os.system("mv dqmc.out dqmc_rhf_rhf.out")
###############################################


################# RHF/CCSD FP-AFQMC ################  put right=ccsd (here) or right=uccsd. Assumes that the t1 and t2 are written as ccsd.h5 or uccsd.h5 respectively
mycc = cc.CCSD(mf)
mycc.frozen = norbFrozen
mycc.verbose = 5
mycc.kernel()
QMCUtils.write_ccsd(mycc.t1, mycc.t2)  #Use QMCUtils.write_ccsd or QMCUtils.write_uccsd depending on your system

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)

fp_helper.run_fpafqmc(mf,norb_frozen=norbFrozen,vmc_root=vmc_root,right='ccsd',ene0Guess= e_guess, chol_cut=1e-5,eneSteps = [ 30, 40, 50, 60 ],dt = 0.05,errorTargets = [0.7e-3,0.7e-3,0.7e-3,0.7e-3],choleskyThreshold = 1e-3,nsteps =60)
os.system("mv dqmc.out dqmc_rhf_ccsd.out")
####################################################



############### Multislater/CCSD FP-AFQMC ############ Assumes that the multislater determinants are written in dets.bin and CCSD t1/t2 in ccsd.h5
# casscf
norb_act = 8
nelec_act = 10
mc0 = mcscf.CASSCF(mf, norb_act, nelec_act)
mc0.frozen = norbFrozen
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


fp_helper.run_fpafqmc(mc,norb_frozen=norbFrozen,vmc_root=vmc_root,right='ccsd',ene0Guess= e_guess, chol_cut=1e-5,ndets=10000,eneSteps = [ 30, 40, 50, 60 ],dt = 0.05,errorTargets = [0.7e-3,0.7e-3,0.7e-3,0.7e-3],choleskyThreshold = 1e-3,nsteps =60)

#######################################################

