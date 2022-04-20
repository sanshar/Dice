import numpy as np
import math
import time
import h5py
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp
#from pyscf.shciscf import shci, settings
from pyscf import lo
from pyscf.lo import pipek, boys, edmiston, iao, ibo
import sys
from scipy.linalg import fractional_matrix_power
from scipy.stats import ortho_group
import scipy.linalg as la
import json

def doRHF(mol):
  mf = scf.RHF(mol)
  print(mf.kernel())
  return mf

#it may be necessary (more often than not) to provide a system curated initial guess for the dm rather than just adding noise
#if using noise, changing its magnitude may change the final answer as well
def doUHF(mol, dm=None):
  umf = scf.UHF(mol)
  if dm is None:
    dm = umf.get_init_guess()
    norb = mol.nao
    dm[0] = dm[0] + np.random.rand(norb, norb) / 2
    dm[1] = dm[1] + np.random.rand(norb, norb) / 2
  print(umf.kernel(dm0 = dm))
  return umf

#it may be necessary (more often than not) to provide a system curated initial guess for the dm rather than just adding noise
#if using noise, changing its magnitude may change the final answer as well
def doGHF(mol, dm=None):
  gmf = scf.GHF(mol)
  gmf.max_cycle = 200
  if dm is None:
    dm = gmf.get_init_guess()
    norb = mol.nao
    dm = dm + np.random.rand(2*norb, 2*norb) / 3
  print(gmf.kernel(dm0 = dm))
  return gmf

def localizeAllElectron(mf, method="lowdin"):
  if (method == "lowdin"):
    return fractional_matrix_power(mf.get_ovlp(), -0.5).T
  elif (method == "pm"):
    return pipek.PM(mf.mol).kernel(mf.mo_coeff)
  elif (method == "boys"):
    return boys.Boys(mf.mol).kernel(mf.mo_coeff)
  elif (method == "er"):
    return edmiston.ER(mf.mol).kernel(mf.mo_coeff)
  elif (method == "iao"):
    return iao.iao(mf.mol, mf.mo_coeff)
  elif (method == "ibo"):
    a = iao.iao(mf.mol, mf.mo_coeff)
    a = lo.vec_lowdin(a, mf.get_ovlp())
    return ibo.ibo(mf.mol, mf.mo_coeff, iaos=a)

def localizeValence(mf, mo_coeff, method="iao"):
  if (method == "iao"):
    return iao.iao(mf.mol, mo_coeff)
  elif (method == "ibo"):
    a = iao.iao(mf.mol, mo_coeff)
    a = lo.vec_lowdin(a, mf.get_ovlp())
    return ibo.ibo(mf.mol, mo_coeff, iaos=a)
  elif (method == "boys"):
    return boys.Boys(mf.mol).kernel(mo_coeff)
  elif (method == "er"):
    return edmiston.ER(mf.mol).kernel(mo_coeff)

# can be used for all electron, but not recommended
def bestDetValence(mol, lmo, occ, eri, writeToFile=True):
  maxLMOContributers = [ np.argmax(np.abs(lmo[::,i])) for i in range(lmo.shape[1]) ]  # index of the ao contributing the most to an lmo
  atomNumAOs = [ i[1][3] - 1 for i in enumerate(mol.aoslice_nr_by_atom()) ]  # end AO index for each atom in ascending order
  lmoSites = [ [] for i in range(mol.natm) ] #lmo's cetered on each atom
  for i in enumerate(maxLMOContributers):
    lmoSites[np.searchsorted(np.array(atomNumAOs), i[1])].append(i[0])

  bestDet = [0 for i in range(lmo.shape[1])]
  def pair(i):
    return i*(i+1)//2+i
  for i in enumerate(occ):
    if eri.ndim == 2:
      onSiteIntegrals = [ (j, eri[pair(j),pair(j)]) for (n,j) in enumerate(lmoSites[i[0]]) ]
    elif eri.ndim == 1:
      onSiteIntegrals = [ (j, eri[pair(pair(j))]) for (n,j) in enumerate(lmoSites[i[0]]) ]
    onSiteIntegrals.sort(key = lambda tup : tup[1], reverse=True)
    for k in range(i[1][0]):
      bestDet[onSiteIntegrals[k][0]] = '2'
    for l in range(i[1][1]):
      bestDet[onSiteIntegrals[i[1][0] + l][0]] = 'a'
    for m in range(i[1][2]):
      bestDet[onSiteIntegrals[i[1][0] + i[1][1] + m][0]] = 'b'

  bestDetStr = '  '.join(bestDet)
  print('bestDet:  ' + bestDetStr)
  if writeToFile:
    fileh = open("bestDet", 'w')
    fileh.write('1.   ' + bestDetStr + '\n')
    fileh.close()

  return bestDetStr

def writeFCIDUMP(mol, mf, lmo, fname='FCIDUMP'):
  h1 = lmo.T.dot(mf.get_hcore()).dot(lmo)
  eri = ao2mo.kernel(mol, lmo)
  tools.fcidump.from_integrals(fname, h1, eri, mol.nao, mol.nelectron, mf.energy_nuc())

def basisChange(matAO, lmo, ovlp):
  matMO = (matAO.T.dot(ovlp).dot(lmo)).T
  return matMO

def writeMat(mat, fileName, isComplex=False):
  fileh = open(fileName, 'w')
  for i in range(mat.shape[0]):
      for j in range(mat.shape[1]):
        if (isComplex):
          fileh.write('(%16.10e, %16.10e) '%(mat[i,j].real, mat[i,j].imag))
        else:
          fileh.write('%16.10e '%(mat[i,j]))
      fileh.write('\n')
  fileh.close()

def readMat(fileName, shape, isComplex=False):
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

def makeAGPFromRHF(rhfCoeffs):
  norb = rhfCoeffs.shape[0]
  nelec = 2*rhfCoeffs.shape[1]
  diag = np.eye(nelec//2)
  #diag = np.zeros((norb,norb))
  #for i in range(nelec/2):
  #  diag[i,i] = 1.
  pairMat = rhfCoeffs.dot(diag).dot(rhfCoeffs.T)
  return pairMat

def makePfaffFromGHF(ghfCoeffs):
  nelec = ghfCoeffs.shape[1]
  amat = np.full((nelec, nelec), 0.)
  for i in range(nelec//2):
    amat[2 * i + 1, 2 * i] = -1.
    amat[2 * i, 2 * i + 1] = 1.
  pairMat = ghfCoeffs.dot(amat).dot(ghfCoeffs.T)
  return pairMat

def addNoise(mat, isComplex=False):
  if (isComplex):
    randMat = 0.01 * (np.random.rand(mat.shape[0], mat.shape[1]) + 1j * np.random.rand(mat.shape[0], mat.shape[1]))
    return mat + randMat
  else:
    randMat = 0.01 * np.random.rand(mat.shape[0], mat.shape[1])
    return mat + randMat

def prepAllElectron(mol, loc="lowdin", dm=None, writeFcidump=True, writeMOs=True):
  mf = doRHF(mol)
  lmo = localizeAllElectron(mf, loc)
  if writeFcidump:
    writeFCIDUMP(mol, mf, lmo)
  gmf = doGHF(mol, dm)
  overlap = mf.get_ovlp(mol)
  ghfCoeffs = basisChange(gmf.mo_coeff, la.block_diag(lmo, lmo), la.block_diag(overlap, overlap))
  if writeMOs:
    writeMat(ghfCoeffs, "hf.txt", False)

def prepValence(mol, ncore, nact, occ=None, loc="iao", dm=None, writeFcidump=True, writeMolden=False, writeBestDet=True, writeMOs=True):
  mf = doRHF(mol)
  mo = mf.mo_coeff
  lmo = localizeValence(mf, mo[:,ncore:ncore+nact], loc)
  if writeMolden:
    tools.molden.from_mo(mol, 'valenceOrbs.molden', lmo)

  nelec = mol.nelectron - 2 * ncore
  mc = mcscf.CASSCF(mf, nact, nelec)
  h1cas, energy_core = mcscf.casci.h1e_for_cas(mc, mf.mo_coeff, nact, ncore)
  mo_core = mc.mo_coeff[:,:ncore]
  core_dm = 2 * mo_core.dot(mo_core.T)
  corevhf = mc.get_veff(mol, core_dm)
  h1eff = lmo.T.dot(mc.get_hcore() + corevhf).dot(lmo)
  eri = ao2mo.kernel(mol, lmo)
  if writeFcidump:
    tools.fcidump.from_integrals('FCIDUMP', h1eff, eri, nact, nelec, energy_core)
  if occ is not None:
    bestDetValence(mol, lmo, occ, eri, writeBestDet)

  # make fictitious valence only molecule and perform ghf
  norb = nact
  molA = gto.M()
  molA.nelectron = nelec
  molA.verbose = 4
  molA.incore_anyway = True
  gmf = scf.GHF(molA)
  gmf.get_hcore = lambda *args: la.block_diag(h1eff, h1eff)
  gmf.get_ovlp = lambda *args: np.identity(2*norb)
  gmf.energy_nuc = lambda *args: energy_core
  gmf._eri = eri
  if dm is None:
    dm = gmf.get_init_guess()
    dm = dm + 2 * np.random.rand(2*norb, 2*norb)
  gmf.level_shift = 0.1
  gmf.max_cycle = 500
  print(gmf.kernel(dm0 = dm))
  if writeMOs:
    writeMat(gmf.mo_coeff, "hf.txt", False)


# calculate and write cholesky integrals
# mc has to be provided if using frozen core
def prepAFQMC(mol, mf, mc=None, chol_cut=1e-5, verbose=False):
  mo_coeff = mf.mo_coeff
  if mc is not None:
    mo_coeff = mc.mo_coeff.copy()
  h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut, verbose)
  nbasis = h1e.shape[-1]
  nelec = mol.nelec

  if mc is not None:
    if mc.ncore != 0:
      nelec = mc.nelecas
      h1e, enuc = mc.get_h1eff()
      chol = chol.reshape((-1, nbasis, nbasis))
      chol = chol[:, mc.ncore:, mc.ncore:]

  nbasis = h1e.shape[-1]
  print(f'nelec: {nelec}')
  print(f'nbasis: {nbasis}')
  print(f'chol.shape: {chol.shape}')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - v0
  chol = chol.reshape((chol.shape[0], -1))
  write_dqmc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')


# cholesky generation functions are from pauxy
def generate_integrals(mol, hcore, X, chol_cut=1e-5, verbose=False):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    if (len(X.shape) == 2):
        h1e = np.dot(X.T, np.dot(hcore, X))
    elif (len(X.shape) == 3):
        h1e = np.dot(X[0].T, np.dot(hcore, X[0]))

    nbasis = h1e.shape[-1]
    # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
    if verbose:
        print (" # Performing modified Cholesky decomposition on ERI tensor.")
    chol_vecs = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print (" # Orthogonalising Cholesky vectors.")
    start = time.time()
    # Step 2.a Orthogonalise Cholesky vectors.
    if (len(X.shape) == 2):
        ao2mo_chol(chol_vecs, X)
    elif (len(X.shape) == 3):
        ao2mo_chol(chol_vecs, X[0])
    if verbose:
        print (" # Time to orthogonalise: %f"%(time.time() - start))
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec
    return h1e, chol_vecs, nelec, enuc

def ao2mo_chol(eri, C):
    nb = C.shape[-1]
    for i, cv in enumerate(eri):
        half = np.dot(cv.reshape(nb,nb), C)
        eri[i] = np.dot(C.conj().T, half).ravel()

def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao*nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = np.zeros((nchol_max, nao*nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs."%nchol_max)
        print("# max number of cholesky vectors = %d"%nchol_max)
        print("# iteration %5d: delta_max = %f"%(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao*nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph',
                         shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = np.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %5d: delta_max = %13.8e: time = %13.8e"%info)

    return chol_vecs[:nchol]

# write cholesky integrals
def write_dqmc(hcore, hcore_mod, chol, nelec, nmo, enuc, ms=0,
                        filename='FCIDUMP_chol'):
    assert len(chol.shape) == 2
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5['hcore'] = hcore.flatten()
        fh5['hcore_mod'] = hcore_mod.flatten()
        fh5['chol'] = chol.flatten()
        fh5['energy_core'] = enuc


# write soc integrals
def write_dqmc_soc(hcore, hcore_mod, chol, nelec, nmo, enuc, filename='FCIDUMP_chol'):
    assert len(chol.shape) == 2
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, chol.shape[0]])
        fh5['hcore_real'] = hcore.real.flatten()
        fh5['hcore_imag'] = hcore.imag.flatten()
        fh5['hcore_mod_real'] = hcore_mod.real.flatten()
        fh5['hcore_mod_imag'] = hcore_mod.imag.flatten()
        fh5['chol'] = chol.flatten()
        fh5['energy_core'] = enuc


# write ccsd amplitudes
def write_ccsd(singles, doubles, rotation=None, filename='ccsd.h5'):
  doubles = np.transpose(doubles, (0, 2, 1, 3)).reshape((singles.size, singles.size))
  if rotation is None:
    rotation = np.eye(sum(singles.shape))
  with h5py.File(filename, 'w') as fh5:
    fh5['singles'] = singles.flatten()
    fh5['doubles'] = doubles.flatten()
    fh5['rotation'] = rotation.flatten()


# write uccsd amplitudes
# NB: change from pyscf order for doubles { uu, ud, dd } -> { uu, dd, ud }
def write_uccsd(singles, doubles, rotation=None, filename='uccsd.h5'):
  doubles0 = np.transpose(doubles[0], (0, 2, 1, 3)).reshape((singles[0].size, singles[0].size))
  doubles1 = np.transpose(doubles[2], (0, 2, 1, 3)).reshape((singles[1].size, singles[1].size))
  doubles2 = np.transpose(doubles[1], (0, 2, 1, 3)).reshape((singles[0].size, singles[1].size))
  if rotation is None:
    rotation = np.eye(sum(singles[0].shape))
  with h5py.File(filename, 'w') as fh5:
    fh5['singles0'] = singles[0].flatten()
    fh5['singles1'] = singles[1].flatten()
    fh5['doubles0'] = doubles0.flatten()
    fh5['doubles1'] = doubles1.flatten()
    fh5['doubles2'] = doubles2.flatten()
    fh5['rotation'] = rotation.flatten()


def write_afqmc_input(numAct = None, left = "rhf", right = "rhf", ndets = 100, seed = None, dt = 0.005, nsteps = 50, nwalk = 50, stochasticIter = 500, orthoSteps = 20, choleskyThreshold = 2.0e-3, fname = 'afqmc.json'):
  system = { }
  system["integrals"] = "FCIDUMP_chol"
  if numAct is not None:
    system["numAct"] = numAct

  wavefunction = { }
  wavefunction["left"] = f"{left}"
  wavefunction["right"] = f"{right}"
  if left == "multislater":
    wavefunction["determinants"] = "dets.bin"
    wavefunction["ndets"] = ndets

  sampling = { }
  if seed is None:
    seed = np.random.randint(1, 1e6)
  sampling["seed"] = seed
  sampling["phaseless"] = True
  sampling["dt"] = dt
  sampling["nsteps"] = nsteps
  sampling["nwalk"] = nwalk
  sampling["stochasticIter"] = stochasticIter
  sampling["choleskyThreshold"] = choleskyThreshold
  sampling["orthoSteps"] = orthoSteps

  json_input = {"system": system, "wavefunction": wavefunction, "sampling": sampling}
  json_dump = json.dumps(json_input, indent = 2)

  with open(fname, "w") as outfile:
    outfile.write(json_dump)
  return


# for tilted hubbard model
def findSiteInUnitCell(newsite, size, latticeVectors, sites):
  for a in range(-1, 2):
    for b in range(-1, 2):
      newsitecopy = [newsite[0]+a*size*latticeVectors[0][0]+b*size*latticeVectors[1][0], newsite[1]+a*size*latticeVectors[0][1]+b*size*latticeVectors[1][\
1]]
      for i in range(len(sites)):
        if ( abs(sites[i][0] - newsitecopy[0]) <1e-10 and abs(sites[i][1] - newsitecopy[1]) <1e-10):
          return True, i

  return False, -1

# for 2d square hubbard w/ periodic boundary conditions
def findSiteAtRowNCol(row, col, size):
    if(row % 2 == 1):
        return (row - 1) * size + col
    else:
        return row * size - (col - 1)

def findRowNColAtSite(site, size):
    row = (site - 1)//size + 1
    if(row % 2 == 1):
        col = (site - 1) % size + 1
    else:
        col = size - (site - 1) % size
    return [row, col]

def findNeighbors(site, size):
    neighbors = []
    [row, col] = findRowNColAtSite(site, size)
    #up
    if (row == 1): #top edge
        neighbors.append(findSiteAtRowNCol(size, col, size))
    else:
        neighbors.append(findSiteAtRowNCol(row - 1, col, size))
    #left
    if (col == 1): #left edge
        neighbors.append(findSiteAtRowNCol(row, size, size))
    else:
        neighbors.append(findSiteAtRowNCol(row, col - 1, size))
    #down
    if (row == size): #bottom edge
        neighbors.append(findSiteAtRowNCol(1, col, size))
    else:
        neighbors.append(findSiteAtRowNCol(row + 1, col, size))
    #right
    if (col == size): #right edge
        neighbors.append(findSiteAtRowNCol(row, 1, size))
    else:
        neighbors.append(findSiteAtRowNCol(row, col + 1, size))
    return neighbors

if __name__=="__main__":
  # make your molecule here
  atomstring = '''
  C  0.000517 0.000000  0.000299
  C  0.000517 0.000000  1.394692
  C  1.208097 0.000000  2.091889
  C  2.415677 0.000000  1.394692
  C  2.415677 0.000000  0.000299
  C  1.208097 0.000000 -0.696898
  H -0.939430 0.000000 -0.542380
  H -0.939430 0.000000  1.937371
  H  1.208097 0.000000  3.177246
  H  3.355625 0.000000  1.937371
  H  3.355625 0.000000 -0.542380
  H  1.208097 0.000000 -1.782255
  '''
  mol = gto.M(
      atom = atomstring,
      unit = 'angstrom',
      basis = 'sto-6g',
      verbose = 4,
      symmetry= 0,
      spin = 0)

  # valence only example
  # alternating spins on neighboring atoms
  occ = []
  configC = [[0,4,0], [0,0,4]] # no double occ, 4 up or 4 dn
  configH = [[0,1,0], [0,0,1]] # no double occ, 1 up or 1 dn
  for i in range(6):
    occ.append(configC[i%2])
  for i in range(6):
    occ.append(configH[(i+1)%2])
  prepValence(mol, 6, 30, occ, loc="ibo")

  # all electron example
  #configC = ["2 a a a a ", "2 b b b b "]
  #configH = ["a ", "b "]
  #bestDetStr = ""
  #for i in range(6):
  #  bestDetStr += configC[i%2]
  #for i in range(6):
  #  bestDetStr += configH[(i+1)%2]
  #fileh = open("bestDet", 'w')
  #fileh.write('1.   ' + bestDetStr + '\n')
  #fileh.close()
  #prepAllElectron(mol)
