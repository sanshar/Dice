import sys, os, time
import math
import numpy as np
import copy
import h5py, json, csv, struct
import pandas as pd
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp, lo
from pyscf.lo import pipek, boys, edmiston, iao, ibo
from scipy.linalg import fractional_matrix_power
from scipy.stats import ortho_group
import scipy.linalg as la

# vmc

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

def makeAGPFromRHF(rhfCoeffs):
  norb = rhfCoeffs.shape[0]
  nelec = 2*rhfCoeffs.shape[1]
  diag = np.eye(nelec//2)
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

# misc

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

def addNoise(mat, isComplex=False):
  if (isComplex):
    randMat = 0.01 * (np.random.rand(mat.shape[0], mat.shape[1]) + 1j * np.random.rand(mat.shape[0], mat.shape[1]))
    return mat + randMat
  else:
    randMat = 0.01 * np.random.rand(mat.shape[0], mat.shape[1])
    return mat + randMat


# dice

# reading dets from dice
def read_dets(fname = 'dets.bin', ndets = None):
  state = { }
  norbs = 0
  with open(fname, 'rb') as f:
    ndetsAll = struct.unpack('i', f.read(4))[0]
    norbs = struct.unpack('i', f.read(4))[0]
    if ndets is None:
      ndets = ndetsAll
    for i in range(ndets):
      coeff = struct.unpack('d', f.read(8))[0]
      det = [ [ 0 for i in range(norbs) ], [ 0 for i in range(norbs) ] ]
      for j in range(norbs):
        occ = struct.unpack('c', f.read(1))[0]
        if (occ == b'a'):
          det[0][j] = 1
        elif (occ == b'b'):
          det[1][j] = 1
        elif (occ == b'2'):
          det[0][j] = 1
          det[1][j] = 1
      state[tuple(map(tuple, det))] = coeff

  return norbs, state

# a_i^dag a_j
def ci_parity(det, i, j):
  assert(det[i] == 0 and det[j] == 1)
  if i > j:
    return (-1.)**(det[j+1:i].count(1))
  else:
    return (-1.)**(det[i+1:j].count(1))

def calculate_ci_1rdm(norbs, state, ndets=100):
  norm_square = 0.
  counter = 0
  rdm = [ np.zeros((norbs, norbs)), np.zeros((norbs, norbs)) ]
  for det, coeff in state.items():
    counter += 1
    norm_square += coeff * coeff
    det_list = list(map(list, det))
    for sz in range(2):
      occ = [ i for i, x in enumerate(det_list[sz]) if x == 1 ]
      empty = [ i for i, x in enumerate(det_list[sz]) if x == 0 ]
      for j in occ:
        rdm[sz][j, j] += coeff * coeff
        for i in empty:
          new_det = copy.deepcopy(det_list)
          new_det[sz][i] = 1
          new_det[sz][j] = 0
          new_det = tuple(map(tuple, new_det))
          if new_det in state:
            rdm[sz][i, j] += ci_parity(det_list[sz], i, j) * state[new_det] * coeff
    if counter == ndets:
      break

  rdm[0] /= norm_square
  rdm[1] /= norm_square
  return rdm

def write_hci_ghf_uhf_integrals(ham_ints, norb, nelec, tol = 1.e-10, filename='FCIDUMP'):
  enuc = ham_ints['enuc']
  h1g = la.block_diag(ham_ints['h1'][0], ham_ints['h1'][1])
  # arrange orbitals ababab...
  h1g = h1g[:, [ i//2 if (i%2 == 0) else norb + i//2 for i in range(2*norb)]]
  h1g = h1g[[ i//2 if (i%2 == 0) else norb + i//2 for i in range(2*norb)], :]
  eriUp, eriDn, eriUpDn = ham_ints['eri']
  erig = np.zeros((4*norb**2, 4*norb**2))
  for i in range(norb):
    for j in range(norb):
      ij_up = (2*i)*(2*norb)+(2*j)
      ij_dn = (2*i+1)*(2*norb)+(2*j+1)
      ij_uhf = max(i,j) * (max(i,j) + 1) // 2 + min(i,j)
      for k in range(norb):
        for l in range(norb):
          kl_up = (2*k)*(2*norb)+(2*l)
          kl_dn = (2*k+1)*(2*norb)+(2*l+1)
          kl_uhf = max(k,l) * (max(k,l) + 1) // 2 + min(k,l)
          erig[ij_up, kl_up] = eriUp[ij_uhf, kl_uhf]
          erig[ij_up, kl_dn] = eriUpDn[ij_uhf, kl_uhf]
          erig[ij_dn, kl_up] = eriUpDn[kl_uhf, ij_uhf]
          erig[ij_dn, kl_dn] = eriDn[ij_uhf, kl_uhf]

  float_format = '(%16.12e, %16.12e)'
  nsorb = 2*norb
  with open(filename, 'w') as fout:
    # header
    fout.write(' &FCI NORB=%4d,NELEC=%2d,MS2=%d,\n' % (nsorb, nelec, 0))
    fout.write('  ORBSYM=%s\n' % ('1,' * nsorb))
    fout.write('  ISYM=0,\n')
    fout.write(' &END\n')

    # eri
    output_format = float_format + ' %4d %4d %4d %4d\n'
    for i in range(nsorb):
      for j in range(nsorb):
        ij = i*nsorb+j
        for k in range(nsorb):
          for l in range(nsorb):
            kl = k*nsorb+l
            if abs(erig[ij][kl]) > tol:
              fout.write(output_format % (erig[ij][kl], 0., i+1, j+1, k+1, l+1))

    # h1
    output_format = float_format + ' %4d %4d  0  0\n'
    for i in range(nsorb):
      for j in range(nsorb):
        if abs(h1g[i,j]) > tol:
          fout.write(output_format % (h1g[i,j], 0., i+1, j+1))

    # enuc
    output_format = float_format + '  0  0  0  0\n'
    fout.write(output_format % (enuc, 0.0))

  return


# afqmc

# modified cholesky for a give matrix
def modified_cholesky(mat, max_error=1e-6):
    diag = mat.diagonal()
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (mat[nu] - R) / (delta_max)**0.5
        nchol += 1

    return chol_vecs[:nchol]

# calculate and write cholesky integrals
# mc has to be provided if using frozen core
def prepAFQMC(mol, mf, mc=None, chol_cut=1e-5, mo_chol=False):
  mo_coeff = mf.mo_coeff
  if mc is not None:
    mo_coeff = mc.mo_coeff.copy()

  if mo_chol and mc is not None: # generate cholesky in mo basis
    nelec = mc.nelecas
    h1e, enuc = mc.get_h1eff()
    eri = ao2mo.restore(4, mc.get_h2eff(mo_coeff), mc.ncas)
    chol0 = modified_cholesky(eri, chol_cut)
    nchol = chol0.shape[0]
    chol = np.zeros((nchol, mc.ncas, mc.ncas))
    for i in range(nchol):
      for m in range(mc.ncas):
        for n in range(m+1):
          triind = m*(m+1)//2 + n
          chol[i, m, n] = chol0[i, triind]
          chol[i, n, m] = chol0[i, triind]
  else: # generate cholesky in ao basis
    h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut, verbose)
    nbasis = h1e.shape[-1]
    nelec = mol.nelec

    if mc is not None:
      if mc.ncore != 0:
        nelec = mc.nelecas
        h1e, enuc = mc.get_h1eff()
        chol = chol.reshape((-1, nbasis, nbasis))
        chol = chol[:, mc.ncore:mc.ncore + mc.ncas, mc.ncore:mc.ncore + mc.ncas]

  nbasis = h1e.shape[-1]
  print(f'nelec: {nelec}')
  print(f'nbasis: {nbasis}')
  print(f'chol.shape: {chol.shape}')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - v0
  chol = chol.reshape((chol.shape[0], -1))
  write_dqmc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

# calculate and write cholesky integrals
def prepAFQMC_soc(mol, mf, soc, chol_cut=1e-5, verbose=False):
  mo_coeff = mf.mo_coeff
  h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut, verbose)
  nbasis = h1e.shape[-1]
  nelec = mol.nelec
  nbasis = h1e.shape[-1]

  mo_coeff_sp = np.block([ [ mo_coeff, np.zeros((nbasis, nbasis)) ] , [ np.zeros((nbasis, nbasis)), mo_coeff ] ])
  h1e = np.block([ [ h1e, np.zeros((nbasis, nbasis)) ] , [ np.zeros((nbasis, nbasis)), h1e ] ])
  h1e = h1e + mo_coeff_sp.T.dot(soc).dot(mo_coeff_sp)
  print(f'nelec: {nelec}')
  print(f'nbasis: {nbasis}')
  print(f'chol.shape: {chol.shape}')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - np.block([ [ v0, np.zeros((nbasis, nbasis)) ] , [ np.zeros((nbasis, nbasis)), v0 ] ])
  chol = chol.reshape((chol.shape[0], -1))
  write_dqmc_soc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, filename='FCIDUMP_chol')

# calculate and write cholesky-like integrals given eri's
def calculate_write_afqmc_uihf_integrals(ham_ints, norb, nelec, ms = 0, chol_cut = 1e-6, filename = 'FCIDUMP_chol', dm=None):
  block_eri = np.block([[ ham_ints['eri'][0], ham_ints['eri'][2] ], [ ham_ints['eri'][2].T, ham_ints['eri'][1] ]])
  #block_eri = block_eri.round(8)
  evecs = modified_cholesky(block_eri, max_error=chol_cut).T
  nchol = evecs.shape[1]
  print(f'nchol: {nchol}')
  chol = np.zeros((2, nchol, norb, norb))
  for i in range(nchol):
    for m in range(norb):
      for n in range(m+1):
        triind = m*(m+1)//2 + n
        chol[0, i, m, n] = evecs[triind, -i-1]
        chol[0, i, n, m] = evecs[triind, -i-1]
        chol[1, i, m, n] = evecs[norb*(norb+1)//2 + triind, -i-1]
        chol[1, i, n, m] = evecs[norb*(norb+1)//2 + triind, -i-1]

  #evals, evecs = np.linalg.eigh(block_eri)
  #nchol = (evals > chol_cut).nonzero()[0].shape[0]
  #evals_sqrt = np.sqrt(evals[ evals > chol_cut ])
  #chol = np.zeros((2, nchol, norb, norb))
  #for i in range(nchol):
  #  for m in range(norb):
  #    for n in range(m+1):
  #      triind = m*(m+1)//2 + n
  #      chol[0, i, m, n] = evals_sqrt[-i-1] * evecs[triind, -i-1]
  #      chol[0, i, n, m] = evals_sqrt[-i-1] * evecs[triind, -i-1]
  #      chol[1, i, m, n] = evals_sqrt[-i-1] * evecs[norb*(norb+1)//2 + triind, -i-1]
  #      chol[1, i, n, m] = evals_sqrt[-i-1] * evecs[norb*(norb+1)//2 + triind, -i-1]

  # writing afqmc ints
  h1 = np.array(ham_ints['h1']).round(8)
  enuc = round(ham_ints['enuc'], 8)
  v0_up = 0.5 * np.einsum('nik,njk->ij', chol[0], chol[0], optimize='optimal')
  v0_dn = 0.5 * np.einsum('nik,njk->ij', chol[1], chol[1], optimize='optimal')
  h1_mod = [ h1[0] - v0_up, h1[1] - v0_dn ]
  chol_flat = [ chol[0].reshape((nchol, -1)), chol[1].reshape((nchol, -1)) ]
  write_dqmc_uihf(h1, h1_mod, chol_flat, nelec, norb, enuc, ms=ms, filename=filename)

  # can be used to check if the chol_cut is reasonable
  if dm is not None:
    coul = np.einsum('sgpr,spr->g', chol, dm)
    exc = np.einsum('sgpr,spt->sgrt', chol, dm)
    e2 = (np.einsum('g,g->', coul, coul) - np.einsum('sgtr,sgrt->', exc, exc) )/2
    e1 = np.einsum('ij,ji->', h1[0], dm[0]) + np.einsum('ij,ji->', h1[1], dm[1])
    print(f'ene for given 1dm: {enuc + e1 + e2}')


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

def write_dqmc_uihf(hcore, hcore_mod, chol, nelec, nmo, enuc, ms=0,
                        filename='FCIDUMP_chol'):
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, ms, chol[0].shape[0]])
        fh5['hcore_up'] = hcore[0].flatten()
        fh5['hcore_dn'] = hcore[1].flatten()
        fh5['hcore_mod_up'] = hcore_mod[0].flatten()
        fh5['hcore_mod_dn'] = hcore_mod[1].flatten()
        fh5['chol_up'] = chol[0].flatten()
        fh5['chol_dn'] = chol[1].flatten()
        fh5['energy_core'] = enuc

# reads rdm files and calculates one-body observables and stochastic error
def calculate_observables(observables, constants = None, prefix = './'):
  # nobs is the number of observables
  nobs = len(observables)
  norb = observables[0].shape[0]
  if constants is None:
    constants = np.zeros(nobs)
  fcount = 0
  observables_afqmc = [ ]
  weights = [ ]
  for filename in os.listdir(prefix):
    if (filename.startswith(f'rdm_')):
      fcount += 1
      with open(filename) as fh:
        weights.append(float(fh.readline()))
      cols = list(range(norb))
      df = pd.read_csv(filename, delim_whitespace=True, usecols=cols, header=None, skiprows=1)
      rdm_i = df.to_numpy()
      obs_i = constants.copy()
      for n in range(nobs):
        obs_i[n] += np.trace(np.dot(rdm_i, observables[n]))
      observables_afqmc.append(obs_i)

  weights = np.array(weights)
  observables_afqmc = np.array(observables_afqmc)
  obsMean = np.zeros(nobs)
  obsError = np.zeros(nobs)
  v1 = weights.sum()
  v2 = (weights**2).sum()
  for n in range(nobs):
    obsMean[n] = np.multiply(weights, observables_afqmc[:, n]).sum() / v1
    obsError[n] = (np.multiply(weights, (observables_afqmc[:, n] - obsMean[n])**2).sum() / (v1 - v2 / v1) / (fcount - 1))**0.5
  return [ obsMean, obsError ]

# reads rdm files and calculates one-body observables and stochastic error
def calculate_observables_uihf(observables, constants = None, prefix = './'):
  # nobs is the number of observables
  nobs = len(observables)
  norb = observables[0][0].shape[0]
  if constants is None:
    constants = np.zeros(nobs)
  fcount = [0, 0]
  observables_afqmc =[ [ ], [ ] ]
  weights = [ [ ], [ ] ]   # weights for up and down are the same
  for (i, sz) in enumerate(['up', 'dn']):
   for filename in os.listdir(prefix):
     if (filename.startswith(f'rdm_{sz}')):
       fcount[i] += 1
       with open(filename) as fh:
         weights[i].append(float(fh.readline()))
       cols = list(range(norb))
       df = pd.read_csv(filename, delim_whitespace=True, usecols=cols, header=None, skiprows=1)
       rdm_i = df.to_numpy()
       obs_i = constants.copy()
       for n in range(nobs):
         obs_i[n] += np.trace(np.dot(rdm_i, observables[n][i]))
       observables_afqmc[i].append(obs_i)

  fcount = fcount[0]
  weights = np.array(weights[0])
  observables_afqmc = np.array(observables_afqmc[0]) + np.array(observables_afqmc[1])
  obsMean = np.zeros(nobs)
  obsError = np.zeros(nobs)
  v1 = weights.sum()
  v2 = (weights**2).sum()
  for n in range(nobs):
    obsMean[n] = np.multiply(weights, observables_afqmc[:, n]).sum() / v1
    obsError[n] = (np.multiply(weights, (observables_afqmc[:, n] - obsMean[n])**2).sum() / (v1 - v2 / v1) / (fcount - 1))**0.5
  return [ obsMean, obsError ]

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


def write_afqmc_input(numAct = None, numCore = None, soc = None, left = "rhf", right = "rhf", ndets = 100, excitationLevel = None, seed = None, dt = 0.005, nsteps = 50, nwalk = 50, stochasticIter = 500, orthoSteps = 20, choleskyThreshold = 2.0e-3, rdm = False, fname = 'afqmc.json'):
  system = { }
  system["integrals"] = "FCIDUMP_chol"
  if numAct is not None:
    system["numAct"] = numAct
  if numCore is not None:
    system["numCore"] = numCore
  if soc is not None:
    system["soc"] = soc

  wavefunction = { }
  wavefunction["left"] = f"{left}"
  wavefunction["right"] = f"{right}"
  if left == "multislater":
    wavefunction["determinants"] = "dets.bin"
    wavefunction["ndets"] = ndets
    if excitationLevel is not None:
      wavefunction["excitationLevel"] = excitationLevel

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

  printBlock = { }
  if rdm:
    printBlock['writeOneRDM'] = True

  json_input = {"system": system, "wavefunction": wavefunction, "sampling": sampling, "print": printBlock}
  json_dump = json.dumps(json_input, indent = 2)

  with open(fname, "w") as outfile:
    outfile.write(json_dump)
  return


# lattice models

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
