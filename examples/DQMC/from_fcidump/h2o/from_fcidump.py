import numpy as np
import re
import h5py
import json
import struct
import os
import ctypes
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

def write_dqmc(hcore, hcore_mod, chol, nelec, nmo, enuc, ms=0,
                        filename='FCIDUMP_chol'):
    assert len(chol.shape) == 2
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5['hcore'] = hcore.flatten()
        fh5['hcore_mod'] = hcore_mod.flatten()
        fh5['chol'] = chol.flatten()
        fh5['energy_core'] = enuc

def read(filename, molpro_orbsym=True):
    '''Parse FCIDUMP.  Return a dictionary to hold the integrals and
    parameters with keys:  H1, H2, ECORE, NORB, NELEC, MS, ORBSYM, ISYM

    Kwargs:
        molpro_orbsym (bool): Whether the orbsym in the FCIDUMP file is in
            Molpro orbsym convention as documented in
            https://www.molpro.net/info/current/doc/manual/node36.html
            In return, orbsym is converted to pyscf symmetry convention
    '''
    print('Parsing %s' % filename)
    finp = open(filename, 'r')

    data = []
    for i in range(10):
        line = finp.readline().upper()
        data.append(line)
        if '&END' in line:
            break
    else:
        raise RuntimeError('Problematic FCIDUMP header')

    result = {}
    tokens = ','.join(data).replace('&FCI', '').replace('&END', '')
    tokens = tokens.replace(' ', '').replace('\n', '').replace(',,', ',')
    for token in re.split(',(?=[a-zA-Z])', tokens):
        key, val = token.split('=')
        if key in ('NORB', 'NELEC', 'MS2', 'ISYM'):
            result[key] = int(val.replace(',', ''))
        elif key in ('ORBSYM',):
            result[key] = [int(x) for x in val.replace(',', ' ').split()]
        else:
            result[key] = val

    norb = result['NORB']
    norb_pair = norb * (norb+1) // 2
    h1e = np.zeros((norb,norb))
    h2e = np.zeros(norb_pair*(norb_pair+1)//2)
    dat = finp.readline().split()
    while dat:
        i, j, k, l = [int(x) for x in dat[1:5]]
        if k != 0:
            if i >= j:
                ij = i * (i-1) // 2 + j-1
            else:
                ij = j * (j-1) // 2 + i-1
            if k >= l:
                kl = k * (k-1) // 2 + l-1
            else:
                kl = l * (l-1) // 2 + k-1
            if ij >= kl:
                h2e[ij*(ij+1)//2+kl] = float(dat[0])
            else:
                h2e[kl*(kl+1)//2+ij] = float(dat[0])
        elif k == 0:
            if j != 0:
                h1e[i-1,j-1] = float(dat[0])
            else:
                result['ECORE'] = float(dat[0])
        dat = finp.readline().split()

    idx, idy = np.tril_indices(norb, -1)
    if np.linalg.norm(h1e[idy,idx]) == 0:
        h1e[idy,idx] = h1e[idx,idy]
    elif np.linalg.norm(h1e[idx,idy]) == 0:
        h1e[idx,idy] = h1e[idy,idx]
    result['H1'] = h1e
    result['H2'] = h2e
    finp.close()
    return result


#def prepDICE_fromFCIDUMP():
def restore(eri,norb):
    npair = norb*(norb+1)//2
    e = np.zeros((npair,npair))
    for i in range(0,npair+1):
        if(i==0):
            continue
        pri = (i-1)*i//2
        nei = i*(i+1)//2
        s = eri[pri:nei]
        for j in range(0,len(s)):
            e[i-1][j] = s[j]
            e[j][i-1] = s[j]
    return e       

def prepAFQMC_fromFCIDUMP(choleskyThreshold=2.e-3,ndets=100,left=None,right=None,seed = None,norb_core=0,norb_frozen=None,nroot=0,mo_coeff=None,chol_cut = 1e-5,dt = 0.005, steps_per_block = 50, nwalk_per_proc = 5, nblocks = 1000, ortho_steps = 20, cholesky_threshold = 2.0e-3,fname="afqmc.json",fcidump="FCIDUMP"): 
    print("Calculating Cholesky integrals")
    fcidump = read(filename=fcidump,molpro_orbsym=True)
    spin = fcidump['MS2']
    norb = fcidump['NORB']
    nelec = fcidump['NELEC']
    eri = restore(fcidump['H2'],norb )
    h1e = fcidump['H1']
    nbasis = norb
    if(mo_coeff == None):
        mo_coeff = np.eye(norb)
    enuc = fcidump['ECORE']
    chol0 = modified_cholesky(eri, chol_cut)
    nchol = chol0.shape[0]
    chol = np.zeros((nchol, norb, norb))
    for i in range(nchol):
      for m in range(norb):
        for n in range(m+1):
          triind = m*(m+1)//2 + n
          chol[i, m, n] = chol0[i, triind]
          chol[i, n, m] = chol0[i, triind]

    print("Finished calculating Cholesky integrals\n")

    print('Size of the correlation space:')
    #print(f'Number of electrons: {mf.mol.nelec}')
    print(f'Number of basis functions: {nbasis}')
    print(f'Number of Cholesky vectors: {chol.shape[0]}\n')
    chol = chol.reshape((-1, norb, norb))
    v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
    h1e_mod = h1e - v0
    chol = chol.reshape((chol.shape[0], -1))



# write mo coefficients
    
    if(left=='multislater'):
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
        right = hf_type
    else:
        hf_type = 'rhf'
        if (left=='uhf'):
            hf_type = "uhf"    
            rhfCoeffs = np.eye(norb)
            uhfCoeffs = np.block([ rhfCoeffs, rhfCoeffs ])
            writeMat(uhfCoeffs, "uhf.txt")
        elif (left=='rhf'):
            hf_type = "rhf"
            rhfCoeffs = np.eye(norb)
            writeMat(rhfCoeffs, "rhf.txt")
        left = hf_type
        right = hf_type 
    
    write_dqmc(h1e, h1e_mod, chol, nelec, norb, enuc, ms=spin, filename='FCIDUMP_chol')
    if(seed==None):
        seed = np.random.randint(0,1e6)
    if(left=='multislater'):
        write_afqmc_input(seed=seed, left=left,numAct=norb_act,numCore=norb_core,ndets=ndets, right=hf_type,detFile=det_file, choleskyThreshold=choleskyThreshold,dt = dt, nsteps = steps_per_block, nwalk = nwalk_per_proc, stochasticIter = nblocks, orthoSteps = ortho_steps, fname=fname)
    else:
        write_afqmc_input(seed=seed, left=left, right=hf_type, choleskyThreshold=choleskyThreshold,dt = dt, nsteps = steps_per_block, nwalk = nwalk_per_proc, stochasticIter = nblocks, orthoSteps = ortho_steps, fname=fname)
