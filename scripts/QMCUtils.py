import sys, os, time
import math
import numpy as np
import copy
import h5py, json, csv, struct
import pandas as pd
from functools import reduce
from pyscf import gto, scf, ao2mo, mcscf, tools, fci, mp, lo, __config__ , lib
from pyscf.lib import logger
from pyscf.lo import pipek, boys, edmiston, iao, ibo
from pyscf.shciscf import shci
from pyscf.ao2mo import _ao2mo
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

  return norbs, state, ndetsAll

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

def calc_uhf_integrals(umf, norb=None, nelec=None):
  if norb is None:
    norb = umf.mol.nao
  if nelec is None:
    nelec = umf.mol.nelectron
  h1_ao = umf.get_hcore()
  h1 = [ umf.mo_coeff[0].T.dot(h1_ao).dot(umf.mo_coeff[0]), umf.mo_coeff[1].T.dot(h1_ao).dot(umf.mo_coeff[1]) ]
  eriUp = ao2mo.kernel(umf._eri, umf.mo_coeff[0])
  eriDn = ao2mo.kernel(umf._eri, umf.mo_coeff[1])
  eriUpDn = ao2mo.incore.general(umf._eri, (umf.mo_coeff[0], umf.mo_coeff[0], umf.mo_coeff[1],umf.mo_coeff[1]))
  enuc = umf.energy_nuc()
  ham_ints = {'enuc': enuc, 'h1': h1, 'eri': [ eriUp, eriDn, eriUpDn ] }
  return ham_ints

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
    h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut)
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


# calculate and write cholesky integrals
def prepAFQMC_gihf(mol, gmf, chol_cut=1e-5):
  norb = mol.nao
  chol_vecs = chunked_cholesky(mol, max_error=1e-5)
  nchol = chol_vecs.shape[0]
  chol = np.zeros((nchol, 2*norb, 2*norb))
  for i in range(nchol):
    chol_i = chol_vecs[i].reshape(norb, norb)
    chol_i = la.block_diag(chol_i, chol_i)
    chol[i] = gmf.mo_coeff.T.dot(chol_i).dot(gmf.mo_coeff)
  hcore = gmf.get_hcore()
  h1e = gmf.mo_coeff.T.dot(hcore).dot(gmf.mo_coeff)
  enuc = mol.energy_nuc()
  nbasis = h1e.shape[-1]
  print(f'nelec: {mol.nelec}')
  print(f'nbasis: {nbasis}')
  print(f'chol.shape: {chol.shape}')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - v0
  chol = chol.reshape((chol.shape[0], -1))
  write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

def run_afqmc(mf_or_mc, vmc_root = None, mpi_prefix = None, mo_coeff = None, ndets = 100, nroot = 0, norb_frozen = 0, nproc = None, chol_cut = 1e-5, seed = None, dt = 0.005, steps_per_block = 50, nwalk_per_proc = 5, nblocks = 1000, ortho_steps = 20, burn_in = 50, cholesky_threshold = 2.0e-3, weight_cap = None, write_one_rdm = False, run_dir = None, scratch_dir = None):
  if isinstance(mf_or_mc, (scf.rhf.RHF, scf.uhf.UHF)):
    run_afqmc_mf(mf_or_mc, vmc_root = vmc_root, mpi_prefix = mpi_prefix, mo_coeff = mo_coeff, norb_frozen = norb_frozen, nproc = nproc, chol_cut = chol_cut, seed = seed, dt = dt, steps_per_block = steps_per_block, nwalk_per_proc = nwalk_per_proc, nblocks = nblocks, ortho_steps = ortho_steps, burn_in = burn_in, cholesky_threshold = cholesky_threshold, weight_cap = weight_cap, write_one_rdm = write_one_rdm, run_dir = run_dir, scratch_dir = scratch_dir)
  elif isinstance(mf_or_mc, mcscf.mc1step.CASSCF):
    run_afqmc_mc(mf_or_mc, vmc_root = vmc_root, mpi_prefix = mpi_prefix, ndets = ndets, nroot = nroot, norb_frozen = norb_frozen, nproc = nproc, chol_cut = chol_cut, seed = seed, dt = dt, steps_per_block = steps_per_block, nwalk_per_proc = nwalk_per_proc, nblocks = nblocks, ortho_steps = ortho_steps, burn_in = burn_in, cholesky_threshold = cholesky_threshold, weight_cap = weight_cap, write_one_rdm = write_one_rdm, run_dir = run_dir, scratch_dir = scratch_dir)
  else:
    raise Exception("Need either mean field or casscf object!")

# performs phaseless afqmc with mf trial
def run_afqmc_mf(mf, vmc_root = None, mpi_prefix = None, mo_coeff = None, norb_frozen = 0, nproc = None, chol_cut = 1e-5, seed = None, dt = 0.005, steps_per_block = 50, nwalk_per_proc = 5, nblocks = 1000, ortho_steps = 20, burn_in = 50, cholesky_threshold = 2.0e-3, weight_cap = None, write_one_rdm = False, run_dir = None, scratch_dir = None):
  print("\nPreparing AFQMC calculation")
  if vmc_root is None:
    vmc_root = os.environ['VMC_ROOT']

  owd = os.getcwd()
  if run_dir is not None:
    os.system(f"rm -rf {run_dir}; mkdir -p {run_dir};")
    os.chdir(f'{run_dir}')
    if scratch_dir is not None:
      os.system(f"mkdir -p {scratch_dir};")

  mol = mf.mol
  # choose the orbital basis
  if mo_coeff is None:
    if isinstance(mf, scf.uhf.UHF):
      mo_coeff = mf.mo_coeff[0]
    elif isinstance(mf, scf.rhf.RHF):
      mo_coeff = mf.mo_coeff
    else:
      raise Exception("Invalid mean field object!")

  # calculate cholesky integrals
  print("Calculating Cholesky integrals")
  h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut)
  nbasis = h1e.shape[-1]
  nelec = mol.nelec

  if norb_frozen > 0:
    assert(norb_frozen*2 < sum(nelec))
    mc = mcscf.CASSCF(mf, mol.nao-norb_frozen, mol.nelectron-2*norb_frozen)
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()
    chol = chol.reshape((-1, nbasis, nbasis))
    chol = chol[:, mc.ncore:mc.ncore + mc.ncas, mc.ncore:mc.ncore + mc.ncas]

  print("Finished calculating Cholesky integrals\n")

  nbasis = h1e.shape[-1]
  print('Size of the correlation space:')
  print(f'Number of electrons: {nelec}')
  print(f'Number of basis functions: {nbasis}')
  print(f'Number of Cholesky vectors: {chol.shape[0]}\n')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - v0
  chol = chol.reshape((chol.shape[0], -1))

  write_dqmc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

  # write mo coefficients
  overlap = mf.get_ovlp(mol)
  if isinstance(mf, (scf.uhf.UHF, scf.rohf.ROHF)):
    hf_type = "uhf"
    uhfCoeffs = np.empty((nbasis, 2*nbasis))
    if isinstance(mf, scf.uhf.UHF):
      q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[0][:, norb_frozen:]))
      uhfCoeffs[:, :nbasis] = q
      q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[1][:, norb_frozen:]))
      uhfCoeffs[:, nbasis:] = q
    else:
      q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
      uhfCoeffs[:, :nbasis] = q
      uhfCoeffs[:, nbasis:] = q

    writeMat(uhfCoeffs, "uhf.txt")

  elif isinstance(mf, scf.rhf.RHF):
    hf_type = "rhf"
    q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
    writeMat(q, "rhf.txt")

  # write input
  write_afqmc_input(seed = seed, left = hf_type, right = hf_type, dt = dt, nsteps = steps_per_block, nwalk = nwalk_per_proc, stochasticIter = nblocks, orthoSteps = ortho_steps, burnIter = burn_in, choleskyThreshold = cholesky_threshold, weightCap = weight_cap, writeOneRDM = write_one_rdm)

  print(f"Starting AFQMC / MF calculation", flush=True)
  e_afqmc = None
  err_afqmc = None
  if mpi_prefix is None:
    mpi_prefix = "mpirun "
    if nproc is not None:
      mpi_prefix += f" -np {nproc} "
  os.system("export OMP_NUM_THREADS=1; rm -f samples.dat")
  afqmc_binary = vmc_root + "/bin/DQMC"

  command = f"{mpi_prefix} {afqmc_binary} afqmc.json"
  os.system(command)

  if (os.path.isfile('samples.dat')):
    print("\nBlocking analysis:", flush=True)
    command = f"mv blocking.tmp blocking.out; cat blocking.out"
    os.system(command)
    print(f"Finished AFQMC / MF calculation\n", flush=True)

    # get afqmc energy from output
    with open(f'blocking.out', 'r') as fh:
      for line in fh:
        if 'Mean energy:' in line:
          ls = line.split()
          e_afqmc = float(ls[2])
        if 'Stochastic error estimate:' in line:
          ls = line.split()
          err_afqmc = float(ls[3])

    if err_afqmc is not None:
      sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
      sig_err = np.around(np.round(err_afqmc * 10**sig_dec) * 10**(-sig_dec), sig_dec)
      sig_e = np.around(e_afqmc, sig_dec)
      print(f'AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n')
    elif e_afqmc is not None:
      print(f'AFQMC energy: {e_afqmc}\nCould not find a stochastic error estimate, check blocking analysis\n', flush=True)

  else:
    print("\nAFQMC calculation did not finish, check the afqmc.dat file\n")
    exit(1)

  os.chdir(owd)

  return e_afqmc, err_afqmc


# performs phaseless afqmc with hci trial
def run_afqmc_mc(mc, vmc_root = None, mpi_prefix = None, norb_frozen = 0, nproc = None, chol_cut = 1e-5, ndets = 100, nroot = 0, seed = None, dt = 0.005, steps_per_block = 50, nwalk_per_proc = 5, nblocks = 1000, ortho_steps = 20, burn_in = 50, cholesky_threshold = 2.0e-3, weight_cap = None, write_one_rdm = False, run_dir = None, scratch_dir = None):
  print("\nPreparing AFQMC calculation")
  if vmc_root is None:
    vmc_root = os.environ['VMC_ROOT']

  owd = os.getcwd()
  if run_dir is not None:
    os.system(f"rm -rf {run_dir}; mkdir -p {run_dir}; cp dets*.bin {run_dir}")
    os.chdir(f'{run_dir}')
    if scratch_dir is not None:
      os.system(f"mkdir -p {scratch_dir}")

  mol = mc.mol
  mo_coeff = mc.mo_coeff

  # calculate cholesky integrals
  print("Calculating Cholesky integrals")
  h1e, chol, nelec, enuc = generate_integrals(mol, mc.get_hcore(), mo_coeff, chol_cut)
  nbasis = h1e.shape[-1]
  nelec = mol.nelec

  # norb_frozen: not correlated in afqmc, only contribute to 1-body potential
  # norb_core: correlated in afqmc but not in trial hci state
  norb_core = int(mc.ncore)

  if norb_frozen > 0:
    assert(norb_frozen*2 < sum(nelec))
    mc_dummy = mcscf.CASSCF(mol, mol.nao-norb_frozen, mol.nelectron-2*norb_frozen)
    norb_core -= norb_frozen
    nelec = mc_dummy.nelecas
    mc_dummy.mo_coeff = mo_coeff
    h1e, enuc = mc_dummy.get_h1eff()
    chol = chol.reshape((-1, nbasis, nbasis))
    chol = chol[:, mc_dummy.ncore:mc_dummy.ncore + mc_dummy.ncas, mc_dummy.ncore:mc_dummy.ncore + mc_dummy.ncas]
    nbasis = h1e.shape[-1]

  print("Finished calculating Cholesky integrals\n")

  print('Size of the correlation space:')
  print(f'Number of electrons: {nelec}')
  print(f'Number of basis functions: {nbasis}')
  print(f'Number of Cholesky vectors: {chol.shape[0]}\n')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - v0
  chol = chol.reshape((chol.shape[0], -1))

  write_dqmc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

  # write mo coefficients
  det_file = 'dets.bin'
  if nroot > 0:
    det_file = f'dets_{nroot}.bin'
  norb_act, state, ndets_all = read_dets(det_file, 1)
  up = np.argsort(-np.array(list(state.keys())[0][0])) + norb_core
  dn = np.argsort(-np.array(list(state.keys())[0][1])) + norb_core + nbasis
  hf_type = "rhf"
  if list(state.keys())[0][0] == list(state.keys())[0][1]:
    rhfCoeffs = np.eye(nbasis)
    writeMat(rhfCoeffs[:, np.concatenate((range(norb_core), up, range(norb_core+norb_act, nbasis))).astype(int)], "rhf.txt")
  else:
    hf_type = "uhf"
    uhfCoeffs = np.hstack((np.eye(nbasis), np.eye(nbasis)))
    writeMat(uhfCoeffs[:, np.concatenate((np.array(range(norb_core)), up, np.array(range(norb_core+norb_act, nbasis+norb_core)), dn, np.array(range(nbasis+norb_core+norb_act, 2*nbasis)))).astype(int)], "uhf.txt")

  # determine ndets for doing calculations
  ndets_list = [ ]
  if isinstance(ndets, list):
    for i, n in enumerate(ndets):
      if n < ndets_all:
        ndets_list.append(n)
    if len(ndets_list) == 0:
      ndets_list = [ ndets_all ]
  elif isinstance(ndets, int):
    if ndets > ndets_all:
      ndets = ndets_all
    ndets_list = [ ndets ]
  else:
    raise Exception('Provide ndets as an int or a list of ints!')

  # run afqmc
  if mpi_prefix is None:
    mpi_prefix = "mpirun "
    if nproc is not None:
      mpi_prefix += f" -np {nproc} "
  os.system("export OMP_NUM_THREADS=1; rm -f samples.dat")
  afqmc_binary = vmc_root + "/bin/DQMC"
  e_afqmc = [ None for _ in ndets_list ]
  err_afqmc = [ None for _ in ndets_list ]
  for i, n in enumerate(ndets_list):
    write_afqmc_input(seed = seed, numAct = norb_act, numCore = norb_core, left = 'multislater', right = hf_type, ndets = n, detFile = det_file, dt = dt, nsteps = steps_per_block, nwalk = nwalk_per_proc, stochasticIter = nblocks, orthoSteps = ortho_steps, burnIter = burn_in, choleskyThreshold = cholesky_threshold, weightCap = weight_cap, writeOneRDM = write_one_rdm, fname = f"afqmc_{n}.json")

    print(f"Starting AFQMC / HCI ({n} dets) calculation", flush=True)
    command = f"{mpi_prefix} {afqmc_binary} afqmc_{n}.json"
    os.system(command)
    if (os.path.isfile('samples.dat')):
      print("\nBlocking analysis:", flush=True)
      command = f"mv samples.dat samples_{n}.dat; mv afqmc.dat afqmc_{n}.dat; mv blocking.tmp blocking_{n}.out; cat blocking_{n}.out"
      os.system(command)
      print(f"Finished AFQMC / HCI ({n} dets) calculation\n", flush=True)

      # get afqmc energy from output
      with open(f'blocking_{n}.out', 'r') as fh:
        for line in fh:
          if 'Mean energy:' in line:
            ls = line.split()
            e_afqmc[i] = float(ls[2])
          if 'Stochastic error estimate:' in line:
            ls = line.split()
            err_afqmc[i] = float(ls[3])

      if err_afqmc[i] is not None:
        sig_dec = int(abs(np.floor(np.log10(err_afqmc[i]))))
        sig_err = np.around(np.round(err_afqmc[i] * 10**sig_dec) * 10**(-sig_dec), sig_dec)
        sig_e = np.around(e_afqmc[i], sig_dec)
        print(f'AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n')
      elif e_afqmc[i] is not None:
        print(f'AFQMC energy: {e_afqmc[i]}\nCould not find a stochastic error estimate, check blocking analysis\n', flush=True)
    else:
      print("\nAFQMC calculation did not finish, check the afqmc.dat file\n")
      exit(1)

  os.chdir(owd)

  return e_afqmc, err_afqmc


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
      pre_filename = prefix + '/' + filename
      with open(pre_filename) as fh:
        weights.append(float(fh.readline()))
      cols = list(range(norb))
      df = pd.read_csv(pre_filename, delim_whitespace=True, usecols=cols, header=None, skiprows=1)
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
  observables_afqmc = [ [ ], [ ] ]
  weights = [ [ ], [ ] ]   # weights for up and down are the same
  for (i, sz) in enumerate(['up', 'dn']):
    for filename in os.listdir(prefix):
      if (filename.startswith(f'rdm_{sz}')):
        fcount[i] += 1
        pre_filename = prefix + '/' + filename
        with open(pre_filename) as fh:
          weights[i].append(float(fh.readline()))
        cols = list(range(norb))
        df = pd.read_csv(pre_filename, delim_whitespace=True, usecols=cols, header=None, skiprows=1)
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


def write_afqmc_input(numAct = None, numCore = None, soc = None, intType = None, left = "rhf", right = "rhf", ndets = 100, detFile = 'dets.bin', excitationLevel = None, seed = None, dt = 0.005, nsteps = 50, nwalk = 50, stochasticIter = 500, orthoSteps = 20, burnIter = None, choleskyThreshold = 2.0e-3, weightCap = None, writeOneRDM = False, scratchDir = None, fname = 'afqmc.json'):
  system = { }
  system["integrals"] = "FCIDUMP_chol"
  if numAct is not None:
    system["numAct"] = numAct
  if numCore is not None:
    system["numCore"] = numCore
  if soc is not None:
    system["soc"] = soc
  if intType is not None:
    system["intType"] = intType

  wavefunction = { }
  wavefunction["left"] = f"{left}"
  wavefunction["right"] = f"{right}"
  if left == "multislater":
    wavefunction["determinants"] = detFile
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
  if burnIter is not None:
    sampling["burnIter"] = burnIter
  if weightCap is not None:
    sampling["weightCap"] = weightCap

  printBlock = { }
  if writeOneRDM:
    printBlock["writeOneRDM"] = True
  if scratchDir is not None:
    printBlock["scratchDir"] = scratchDir
  elif writeOneRDM:
    printBlock["scratchDir"] = "rdm"

  json_input = {"system": system, "wavefunction": wavefunction, "sampling": sampling, "print": printBlock}
  json_dump = json.dumps(json_input, indent = 2)
  #print(f"AFQMC input options:\n{json_dump}\n")

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
  

def from_mc(mc, filename, nFrozen=0, orbsym=None,tol=getattr(__config__, 'fcidump_write_tol', 1e-15), float_format=getattr(__config__, 'fcidump_float_format', ' %.16g')):
    mol = mc.mol
    nInner = mc.ncore + mc.ncas - nFrozen
    inner = mc.mo_coeff[:,nFrozen:nFrozen+nInner]
    virtual = mc.mo_coeff[:,nFrozen+nInner:]
    mo_coeff = np.concatenate((inner, virtual), 1)
    if orbsym is None:
        orbsym = getattr(mo_coeff, 'orbsym', None)
    if (nFrozen == 0):
      t = mol.intor_symmetric('int1e_kin')
      v = mol.intor_symmetric('int1e_nuc')
      h1e = reduce(np.dot, (mo_coeff.T, t+v, mo_coeff))
      nuc = mol.energy_nuc()
    else:
      frozen = mc.mo_coeff[:,:nFrozen]
      core_dm = 2 * frozen.dot(frozen.T)
      corevhf = mc.get_veff(mol, core_dm)
      hcore = mc.get_hcore()
      nuc = mc.energy_nuc()
      nuc += np.einsum('ij,ji', core_dm, hcore)
      nuc += np.einsum('ij,ji', core_dm, corevhf) * .5
      h1e = mo_coeff.T.dot(hcore + corevhf).dot(mo_coeff)

    iiii = ao2mo.outcore.general_iofree(mol, (inner,)*4)
    iiiv = ao2mo.outcore.general_iofree(mol, (virtual, inner, inner, inner))
    iviv = ao2mo.outcore.general_iofree(mol, (virtual, inner, virtual, inner))
    iivv = ao2mo.outcore.general_iofree(mol, (virtual, virtual, inner, inner))
    from_integrals_nevpt(filename, h1e, iiii, iiiv, iviv, iivv, h1e.shape[0], mol.nelectron - 2*nFrozen, nuc, mol.ms, orbsym,
                   tol, float_format)


def from_integrals_nevpt(filename, h1e, iiii, iiiv, iviv, iivv, nmo, nelec, nuc=0, ms=0, orbsym=None,tol=getattr(__config__, 'fcidump_write_tol', 1e-15), float_format=getattr(__config__, 'fcidump_float_format', ' %.16g')): 
  fh = h5py.File(filename, 'w')
  header = np.array([nelec, nmo, ms])
  fh['header'] = header
  fh['hcore'] = h1e
  fh['iiii'] = lib.pack_tril(iiii)
  fh['iiiv'] = iiiv.flatten()
  fh['iviv'] = lib.pack_tril(iviv)
  fh['iivv'] = iivv.flatten()
  fh['energy_core'] = nuc
  fh.close()


def run_nevpt2(mc,nelecAct=None,numAct=None,norbFrozen=None, integrals="FCIDUMP.h5",nproc=None, seed=None, fname="nevpt2.json",foutname='nevpt2.out',spatialRDMfile="spatialRDM.0.0.txt",spinRDMfile='',stochasticIterNorms= 1000,nIterFindInitDets= 100,numSCSamples= 10000,stochasticIterEachSC= 100,fixedResTimeNEVPT_Ene= False,epsilon= 1.0e-8,efficientNEVPT_2= True,determCCVV= True,SCEnergiesBurnIn= 50,SCNormsBurnIn= 50,vmc_root=None, diceoutfile="dice.out"):
	
	numCore = (sum(mc.mol.nelec)-nelecAct - norbFrozen*2)//2
	getDets(fname=diceoutfile)
	
	run_ICPT(mc,nelecAct=nelecAct,norbAct=numAct,vmc_root=vmc_root,fname=spatialRDMfile) 
	
	print("Writing NEVPT2 input")
#	DEFAULT_FLOAT_FORMAT = getattr(__config__, 'fcidump_float_format', ' %.16g')
#	TOL = getattr(__config__, 'fcidump_write_tol', 1e-15)
	
	from_mc(mc, 'FCIDUMP.h5', nFrozen = norbFrozen)  #Uses fuctions from pyscf-tools
	write_nevpt2_input(numAct = numAct , numCore = numCore , determinants = 'dets', integrals=integrals ,seed=seed, fname=fname, stochasticIterNorms = stochasticIterNorms, nIterFindInitDets = nIterFindInitDets , numSCSamples = numSCSamples, stochasticIterEachSC = stochasticIterEachSC, fixedResTimeNEVPT_Ene =fixedResTimeNEVPT_Ene , epsilon = epsilon, efficientNEVPT_2 = efficientNEVPT_2, determCCVV = determCCVV , SCEnergiesBurnIn = SCEnergiesBurnIn , SCNormsBurnIn = SCNormsBurnIn)
	fileh = open("moEne.txt", 'w')
	for i in range(mc.mol.nao - norbFrozen):
		fileh.write('%.12e\n'%(mc.mo_energy[i + norbFrozen]))
	fileh.close()
	print("Running NEVPT2")
	if vmc_root is None:
		vmc_root = os.environ['VMC_ROOT']
	vmc_binary=vmc_root+"/bin/VMC"
	mpi_prefix = "mpirun "
	if nproc is not None:
		mpi_prefix += f" -np {nproc} "
	os.system(f"{mpi_prefix} {vmc_binary} {fname} > {foutname}")
	energy,error = get_nevptEnergy(fname=foutname,printNevpt2=True)
	print(f"Total Energy (including CCAV,CCVV,ACVV) = {energy} +/- {error}")

def write_nevpt2_input(numAct=None, numCore=None, determinants="dets", integrals="FCIDUMP.h5", seed=None, fname="nevpt2.json",stochasticIterNorms= 1000,nIterFindInitDets= 100,numSCSamples= 10000,stochasticIterEachSC= 100,fixedResTimeNEVPT_Ene= False,epsilon= 1.0e-8,efficientNEVPT_2= True,determCCVV= True,SCEnergiesBurnIn= 50,SCNormsBurnIn= 50):
	system = {}
	system["integrals"] = integrals
	system["numAct"] = numAct
	system["numCore"] = numCore
	
	prints = {}
	prints["readSCNorms"]= False
	
	wavefunction = {}
	wavefunction["name"]="scpt"
	wavefunction["overlapCutoff"]=1.0e-8
	wavefunction["determinants"] = f"{determinants}"
	
	sampling = {}
	sampling["stochasticIterNorms"] = stochasticIterNorms
	sampling["nIterFindInitDets"] = nIterFindInitDets
	sampling["numSCSamples"] = numSCSamples
	sampling["stochasticIterEachSC"] = stochasticIterEachSC
	sampling["fixedResTimeNEVPT_Ene"] = fixedResTimeNEVPT_Ene
	sampling["epsilon"] = epsilon
	if seed==None:
	        sampling["seed"] = np.random.randint(1,1e6)
	else:
	        sampling["seed"] = seed
	sampling["efficientNEVPT_2"] = efficientNEVPT_2
	sampling["determCCVV"] = determCCVV
	sampling["SCEnergiesBurnIn"] = SCEnergiesBurnIn
	sampling["SCNormsBurnIn"] = SCNormsBurnIn
	
	json_input = {"system": system, "print": prints, "wavefunction": wavefunction, "sampling": sampling}
	json_dump = json.dumps(json_input, indent = 2)
	with open(fname, "w") as outfile:
	        outfile.write(json_dump)
	return

def run_ICPT(mc,nelecAct=None,norbAct=None,vmc_root=None,fname="spatialRDM.0.0.txt"):
	import NEVPT2Helper as nev
	print("Running ICPT\n")
	intfolder = "int/"
	os.system("mkdir -p "+intfolder)
	dm2a = np.zeros((norbAct, norbAct, norbAct, norbAct))
	file2pdm = fname 
	file2pdm = file2pdm.encode()  # .encode for python3 compatibility
	shci.r2RDM(dm2a, norbAct, file2pdm)
	dm1 = np.einsum('ikjj->ki', dm2a)
	dm1 /= (nelecAct - 1)
	#dm1, dm2a = mc.fcisolver.make_rdm12(0, mc.ncas, mc.nelecas)
	dm2 = np.einsum('ijkl->ikjl', dm2a)
	np.save(intfolder+"E2.npy", np.asfortranarray(dm2))
	np.save(intfolder+"E1.npy", np.asfortranarray(dm1))
	print ("trace of 2rdm", np.einsum('ijij',dm2))
	print ("trace of 1rdm", np.einsum('ii',dm1))
	nfro = 0
	E1eff = dm1 # for state average
	nev.writeNEVPTIntegrals(mc, dm1, dm2, E1eff, nfro, intfolder)
	nev.write_ic_inputs(mc.nelecas[0]+mc.nelecas[1], mc.ncore, mc.ncas, nfro, mc.nelecas[0]-mc.nelecas[1],'NEVPT2')
	if vmc_root is None:
    		vmc_root = os.environ['VMC_ROOT']
	os.system("export OMP_NUM_THREADS=1")
	icpt_binary = vmc_root + "/bin/ICPT"
	inps = ['ACVV','CCAV','CCVV']
	for inp in inps:
        	command = f"{icpt_binary} NEVPT2_{inp}.inp > {inp.lower()}.out"
        	print(command)
        	os.system(command)
	print("Finished running ICPT\n")	

def getDets(fname="dice.out"): #To get the determinants printed to dice output and write to text file 'dets'
    file = open(fname,'r')
    content = (file.readlines())
    dets = []
    k = -1
    for c in content:
        if c.split(" ")[0]=="Printing" :
            k = content.index(c)
            break
         
    for c in content[(k+3):]:
        if c.split()[0]=='Printing':
            break
        ch =c.split()
        if(float(ch[0])==0):
            continue
        listToStr = ' '.join(map(str, ch))
        dets.append(listToStr+'\n')
    f = open("dets","w")
    f.writelines(dets)
    f.close()


def get_nevptEnergy(fname="nevpt2.out",printNevpt2=False):
    file = open(fname,'r')
    filelines = file.readlines()[-10:]
    Error = 0
    for line in filelines:
        if 'Energy error estimate' in line:
            Error = float(line.split()[-1])
            break
    nevfile = filelines[-5:]
    if(printNevpt2):
        print(''.join(map(str,nevfile)))
    nevptE = float(nevfile[-1].split()[-1])
    icptnames = ['acvv.out','ccav.out']
    icE = []
    for n in icptnames:
        file = open(n,'r')
        e = float((file.readlines()[-1]).split()[-1])
        icE.append(e)
    totalE = nevptE + sum(icE)
    return totalE,Error
  

