import numpy as np
import scipy.linalg as la
from pyscf import gto
import prepVMC

r = 2.5
atomstring = "N 0 0 0; N 0 0 %g"%(r)
mol = gto.M(
    atom = atomstring,
    basis = '6-31g',
    unit = 'bohr',
    verbose=4,
    symmetry=0,
    spin = 0)
mf = prepVMC.doRHF(mol)
lmo = prepVMC.localizeAllElectron(mf, "lowdin")

# sp hybrids to improve the jastrow
hybrid1p = (lmo[::,1] + lmo[::,5])/2**0.5
hybrid1n = (lmo[::,1] - lmo[::,5])/2**0.5
hybrid2p = (lmo[::,2] + lmo[::,8])/2**0.5
hybrid2n = (lmo[::,2] - lmo[::,8])/2**0.5
lmo[::,1] = hybrid1p
lmo[::,5] = hybrid1n
lmo[::,2] = hybrid2p
lmo[::,8] = hybrid2n
hybrid1p = (lmo[::,10] + lmo[::,14])/2**0.5
hybrid1n = (lmo[::,10] - lmo[::,14])/2**0.5
hybrid2p = (lmo[::,11] + lmo[::,17])/2**0.5
hybrid2n = (lmo[::,11] - lmo[::,17])/2**0.5
lmo[::,10] = hybrid1p
lmo[::,14] = hybrid1n
lmo[::,11] = hybrid2p
lmo[::,17] = hybrid2n

prepVMC.writeFCIDUMP(mol, mf, lmo)

# initial det to start vmc run with
fileh = open("bestDet", 'w')
fileh.write('1.   2 2 0 a a a 0 0 0   2 2 0 b b b 0 0 0\n')
fileh.close()

# rhf
overlap = mf.get_ovlp(mol)
rhfCoeffs = prepVMC.basisChange(mf.mo_coeff, lmo, overlap)
prepVMC.writeMat(rhfCoeffs, "rhf.txt")

# agp
theta = rhfCoeffs[::, :mol.nelectron//2]
pairMat = prepVMC.makeAGPFromRHF(theta)
prepVMC.writeMat(pairMat, "agp.txt")

# uhf
dm = [0. * mf.get_init_guess(), 0. * mf.get_init_guess()]
double = [0, 1]
for i in double:
  dm[0][i, i] = 1.
  dm[1][i, i] = 1.
  dm[0][9+i, 9+i] = 1.
  dm[1][9+i, 9+i] = 1.
single = [3, 4, 5]
for i in single:
  dm[0][i, i] = 1.
  dm[1][9+i, 9+i] = 1.
umf = prepVMC.doUHF(mol, dm)
uhfCoeffs = np.empty((mol.nao, 2*mol.nao))
uhfCoeffs[::,:mol.nao] = prepVMC.basisChange(umf.mo_coeff[0], lmo, overlap)
uhfCoeffs[::,mol.nao:] = prepVMC.basisChange(umf.mo_coeff[1], lmo, overlap)
prepVMC.writeMat(uhfCoeffs, "uhf.txt")

# ghf
dmg = la.block_diag(dm[0], dm[1])
gmf = prepVMC.doGHF(mol, dmg)
ghfCoeffs = prepVMC.basisChange(gmf.mo_coeff, la.block_diag(lmo, lmo), la.block_diag(overlap, overlap))
prepVMC.writeMat(ghfCoeffs, "ghf.txt")

# pfaffian
theta = ghfCoeffs[::, :mol.nelectron]
pairMat = prepVMC.makePfaffFromGHF(theta)
pairMat = prepVMC.addNoise(pairMat)
prepVMC.writeMat(pairMat, "pfaffian.txt")

# multi slater
prepVMC.writeFCIDUMP(mol, mf, mf.mo_coeff, 'FCIDUMP_can')
prepVMC.writeMat(la.block_diag(rhfCoeffs, rhfCoeffs), "rghf.txt")
