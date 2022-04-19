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
atomstring = ""
for i in range(10):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(atom=atomstring, basis='sto-6g', unit='bohr', symmetry=0)
mf = scf.RHF(mol)
mf.kernel()
norb = mol.nao

# uhf
dm = [numpy.zeros((norb, norb)), numpy.zeros((norb, norb))]
for i in range(norb//2):
  dm[0][2*i, 2*i] = 1.
  dm[1][2*i+1, 2*i+1] = 1.
umf = scf.UHF(mol)
umf.kernel(dm)

# afqmc
print("Preparing AFQMC calculation")
# write hf wave function coefficients
rhfCoeffs = numpy.eye(mol.nao)
QMCUtils.writeMat(rhfCoeffs, "rhf.txt")
overlap = mf.get_ovlp(mol)
uhfCoeffs = numpy.empty((norb, 2*norb))
uhfCoeffs[:,:norb] = mf.mo_coeff.T.dot(overlap).dot(umf.mo_coeff[0])
uhfCoeffs[:,norb:] = mf.mo_coeff.T.dot(overlap).dot(umf.mo_coeff[1])
QMCUtils.writeMat(uhfCoeffs, "uhf.txt")

# calculate and write cholesky integrals
QMCUtils.prepAFQMC(mol, mf)
QMCUtils.write_afqmc_input(seed = 4321, left="rhf", right="uhf", nwalk=20, stochasticIter=50, choleskyThreshold=1.e-3, fname="afqmc.json")
