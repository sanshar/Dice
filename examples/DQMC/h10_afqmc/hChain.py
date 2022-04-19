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
r = 1.6
atomstring = ""
for i in range(10):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(
    atom=atomstring,
    basis='sto-6g',
    verbose=4,
    unit='bohr',
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = 0
mycc.verbose = 5
mycc.kernel()

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)

# dice

# writing input and integrals
print("\nPreparing Dice calculation")
# dummy shciscf object for specifying options
mc = shci.SHCISCF(mf, mol.nao, mol.nelectron)
mc.mo_coeff = mf.mo_coeff
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
rhfCoeffs = numpy.eye(mol.nao)
QMCUtils.writeMat(rhfCoeffs, "rhf.txt")

# calculate and write cholesky integrals
QMCUtils.prepAFQMC(mol, mf, mc)

# write afqmc input and perform calculation
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

os.system("export OMP_NUM_THREADS=1; rm samples.dat -f")

# rhf trial
QMCUtils.write_afqmc_input(seed = 4321, left="rhf", right="rhf", nwalk=50, stochasticIter=500, choleskyThreshold=1.e-3, fname="afqmc_rhf.json")
print("\nStarting AFQMC / RHF calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_rhf.json > afqmc_rhf.out;
              mv samples.dat samples_rhf.dat
              python {blocking_script} samples_rhf.dat 50 > blocking_rhf.out;
              cat blocking_rhf.out;
           '''
os.system(command)
print("Finished AFQMC / RHF calculation\n")

# hci trial
QMCUtils.write_afqmc_input(seed = 78813, left="multislater", right="rhf", nwalk=50, stochasticIter=300, choleskyThreshold=1.e-3, fname="afqmc_multislater.json")
print("Starting AFQMC / HCI calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_multislater.json > afqmc_multislater.out;
              mv samples.dat samples_multislater.dat
              python {blocking_script} samples_multislater.dat 50 > blocking_multislater.out;
              cat blocking_multislater.out;
           '''
os.system(command)
print("Finished AFQMC / HCI calculation")
