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
mol = gto.M(
    atom="C 0 0 0",
    basis='augccpvtz',
    verbose=4,
    unit='bohr',
    symmetry='dooh',
    spin=2)
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1g':(2,2), 'A1u':(0,0), 'E1ux':(1,0), 'E1uy':(1,0)}
mf.kernel()

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = 1
mycc.verbose = 5
mycc.kernel()

et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)

# casscf
norbFrozen = 1
mc0 = mcscf.CASSCF(mf, 8, 4)
mo = mc0.sort_mo_by_irrep({'A1g': 2, 'A1u': 2, 'E1ux': 2, 'E1uy': 2}, {'A1g': 1})
mc0.frozen = norbFrozen
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
rhfCoeffs = numpy.eye(mol.nao - norbFrozen)
uhfCoeffs = numpy.block([ rhfCoeffs, rhfCoeffs ])
QMCUtils.writeMat(uhfCoeffs, "uhf.txt")

# calculate and write cholesky integrals
# dummy mcsscf for core averaging
mc = mcscf.CASSCF(mf, mol.nao-norbFrozen, mol.nelectron-2*norbFrozen)
mc.mo_coeff = mc0.mo_coeff
QMCUtils.prepAFQMC(mol, mf, mc)

# write afqmc input and perform calculation
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

os.system("export OMP_NUM_THREADS=1; rm samples.dat -f")

# rohf trial
QMCUtils.write_afqmc_input(seed=89649, left="uhf", right="uhf", nwalk=25, stochasticIter=200, choleskyThreshold=2.e-3, fname="afqmc_rohf.json")
print("\nStarting AFQMC / ROHF calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_rohf.json > afqmc_rohf.out;
              mv samples.dat samples_rohf.dat
              python {blocking_script} samples_rohf.dat 50 > blocking_rohf.out;
              cat blocking_rohf.out;
           '''
os.system(command)
print("Finished AFQMC / ROHF calculation\n")

# hci trial
QMCUtils.write_afqmc_input(seed=142108, numAct=8, left="multislater", right="uhf", nwalk=25, stochasticIter=200, choleskyThreshold=2.e-3, fname="afqmc_multislater.json")
print("Starting AFQMC / HCI calculation", flush=True)
command = f'''
              mpirun -np {nproc} {afqmc_binary} afqmc_multislater.json > afqmc_multislater.out;
              mv samples.dat samples_multislater.dat
              python {blocking_script} samples_multislater.dat 50 > blocking_multislater.out;
              cat blocking_multislater.out;
           '''
os.system(command)
print("Finished AFQMC / HCI calculation")
