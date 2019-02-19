import numpy as np
import math
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
#from pyscf.shciscf import shci, settings
from pyscf.lo import pipek, boys
import sys
from scipy.linalg import fractional_matrix_power
from scipy.stats import ortho_group
import scipy.linalg as la

def doRHF(mol):
  mf = scf.RHF(mol)
  print mf.kernel()
  return mf

def localize(mol, mf, method):
  if (method == "lowdin"):
    return fractional_matrix_power(mf.get_ovlp(mol), -0.5).T
  elif (method == "pm"):
    return pipek.PM(mol).kernel(mf.mo_coeff)
  elif (method == "pmLowdin"):
    lowdin = fractional_matrix_power(mf.get_ovlp(mol), -0.5).T
    return pipek.PM(mol).kernel(lowdin)
  elif (method == "boys"):
    return boys.Boys(mol).kernel(mf.mo_coeff)

# only use after lowdin, and for non-minimal bases
def lmoScramble(mol, lmo):
  scrambledLmo = np.full(lmo.shape, 0.)
  nBasisPerAtom = np.full(mol.natm, 0)
  for i in range(len(gto.ao_labels(mol))):
    nBasisPerAtom[int(gto.ao_labels(mol)[i].split()[0])] += 1
  print nBasisPerAtom
  for i in range(mol.natm):
    n = nBasisPerAtom[i]
    orth = ortho_group.rvs(dim=n)
    scrambledLmo[::,n*i:n*(i+1)] = lmo[::,n*i:n*(i+1)].dot(orth)
  return scrambledLmo

def writeFCIDUMP(mol, mf, lmo):
  h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
  eri = ao2mo.kernel(mol, lmo)
  tools.fcidump.from_integrals('FCIDUMP', h1, eri, mol.nao, mol.nelectron, mf.energy_nuc())

def basisChange(matAO, lmo, ovlp):
  matMO = (matAO.T.dot(ovlp).dot(lmo)).T
  return matMO

def writeMat(mat, fileName, isComplex):
  fileh = open(fileName, 'w')
  for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        if (isComplex):
          fileh.write('(%16.10e, %16.10e) '%(mat[i,j].real, mat[i,j].imag))
        else:
          fileh.write('%16.10e '%(mat[i,j]))
      fileh.write('\n')
  fileh.close()

def readMat(fileName, shape, isComplex):
  if(isComplex):
    matr = np.zeros(shape)
    mati = np.zeros(shape)
  else:
    mat = np.zeros(shape)
  row = 0
  fileh = open(fileName, 'r')
  for line in fileh:
    col = 0
    for coeff in line.split():
      if (isComplex):
        m = coeff.strip()[1:-1]
        matr[row, col], mati[row, col] = [float(x) for x in m.split(',')]
      else:
        mat[row, col]  = float(coeff)
      col = col + 1
    row = row + 1
  fileh.close()
  if (isComplex):
    mat = matr + 1j * mati
  return mat

def doGHF(mol):
  gmf = scf.GHF(mol)
  gmf.max_cycle = 200
  dm = gmf.get_init_guess()
  norb = mol.nao
  dm = dm + np.random.rand(2*norb, 2*norb) / 3
  print gmf.kernel(dm0 = dm)
  return gmf

def makeAGPFromRHF(rhfCoeffs):
  norb = rhfCoeffs.shape[0]
  nelec = 2*rhfCoeffs.shape[1]
  diag = np.eye(nelec/2)
  #diag = np.zeros((norb,norb))
  #for i in range(nelec/2):
  #  diag[i,i] = 1.
  pairMat = rhfCoeffs.dot(diag).dot(rhfCoeffs.T)
  return pairMat

def makePfaffFromGHF(ghfCoeffs):
  nelec = ghfCoeffs.shape[1]
  amat = np.full((nelec, nelec), 0.)
  for i in range(nelec/2):
    amat[2 * i + 1, 2 * i] = -1.
    amat[2 * i, 2 * i + 1] = 1.
  pairMat = ghfCoeffs.dot(amat).dot(ghfCoeffs.T)
  return pairMat

def addNoise(mat, isComplex):
  if (isComplex):
    randMat = 0.01 * (np.random.rand(mat.shape[0], mat.shape[1]) + 1j * np.random.rand(mat.shape[0], mat.shape[1]))
    return mat + randMat
  else:
    randMat = 0.01 * np.random.rand(mat.shape[0], mat.shape[1])
    return mat + randMat

# make your molecule here
r = 2 * 0.529177
atomstring = ""
for i in range(4):
  atomstring += "H 0 0 %g\n"%(i*r)
mol = gto.M(
    atom = atomstring,
    basis = 'sto-6g',
    verbose=4,
    symmetry=0,
    spin = 0)
mf = doRHF(mol)
lmo = localize(mol, mf, "lowdin")
#lmo = lmoScramble(mol, lmo) # use for non-minimal bases
writeFCIDUMP(mol, mf, lmo)
overlap = mf.get_ovlp(mol)
rhfCoeffs = basisChange(mf.mo_coeff, lmo, overlap)
writeMat(rhfCoeffs, "rhf.txt", False)
theta = rhfCoeffs[::, :mol.nelectron/2]
pairMat = makeAGPFromRHF(theta)
writeMat(pairMat, "agp.txt", False)
gmf = doGHF(mol)
ghfCoeffs = basisChange(gmf.mo_coeff, la.block_diag(lmo, lmo), la.block_diag(overlap, overlap))
writeMat(ghfCoeffs, "ghf.txt", False)
theta = ghfCoeffs[::, :mol.nelectron]
pairMat = makePfaffFromGHF(theta)
writeMat(pairMat, "pfaffian.txt", False)
