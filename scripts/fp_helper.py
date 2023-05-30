import QMCUtils
import numpy as np
from pyscf import gto, scf,mcscf,cc
import os
import json

def write_fpafqmc_input(left='rhf',right='rhf',seed=None,dt=0.05,nsteps=60,eneSteps= [ 20, 40, 50, 60 ],errorTargets=[ 0.8e-3, 0.8e-3, 0.8e-3, 0.8e-3 ],choleskyThreshold=0.5e-4,orthoSteps=5,ene0Guess=None,stochasticIter=100000,numAct=None,numCore=None,soc=None,intType=None,ciThreshold=None,filename='dqmc.json',ndets=10000,detFile='dets.bin',excitationLevel=None):
    if(ene0Guess is None): print("Provide the energy guess")
    if(seed is None): seed = np.random.randint(1, 1e6)
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
    if ciThreshold is not None:
        system["ciThreshold"] = ciThreshold
    wavefunction = { }
    wavefunction["left"] = f"{left}"
    wavefunction["right"] = f"{right}"
    if left == "multislater":
    	wavefunction["determinants"] = detFile
    	wavefunction["ndets"] = ndets
    	if excitationLevel is not None:
    		wavefunction["excitationLevel"] = excitationLevel
    
    sampling = {}
    sampling["seed"] = seed
    sampling["dt"] = dt
    sampling["nsteps"] = nsteps
    sampling["eneSteps"] = eneSteps
    sampling["errorTargets"] = errorTargets
    sampling["choleskyThreshold"] = choleskyThreshold
    sampling["orthoSteps"] = orthoSteps
    sampling["ene0Guess"] = ene0Guess
    sampling["stochasticIter"] = stochasticIter
    
    printBlock = { }
    json_input = {"system": system, "wavefunction": wavefunction, "sampling": sampling, "print": printBlock}
    json_dump = json.dumps(json_input, indent = 2)
    fname=filename
    with open(fname, "w") as outfile:
    	outfile.write(json_dump)
    return

def run_fpafqmc(mf_or_mc,norb_frozen=0,mo_coeff=None,chol_cut=1e-5,vmc_root=None,ene0Guess=None,mpi_prefix=None,nproc=None,seed = None,dt = 0.05,nsteps=60,eneSteps= [ 20, 40, 50, 60 ],errorTargets=[ 0.8e-3, 0.8e-3, 0.8e-3, 0.8e-3 ],choleskyThreshold=0.5e-4,orthoSteps=5,stochasticIter=100000,ndets=None,nroot=0,left=None,right=None):
    if isinstance(mf_or_mc, (scf.rhf.RHF, scf.uhf.UHF,scf.rohf.ROHF)):
      return run_fpafqmc_mf(mf_or_mc,norb_frozen=norb_frozen,mo_coeff=mo_coeff,chol_cut=chol_cut,vmc_root=vmc_root,ene0Guess=ene0Guess,mpi_prefix=mpi_prefix,nproc=nproc,seed=seed,dt = dt,nsteps=nsteps,eneSteps= eneSteps,errorTargets=errorTargets,choleskyThreshold=choleskyThreshold,orthoSteps=orthoSteps,stochasticIter=stochasticIter,left=left,right=right)
    else:
      return run_fpafqmc_mc(mf_or_mc,norb_frozen=norb_frozen,mo_coeff=mo_coeff,chol_cut=chol_cut,vmc_root=vmc_root,ene0Guess=ene0Guess,mpi_prefix=mpi_prefix,nproc=nproc,seed=seed,dt = dt,nsteps=nsteps,eneSteps= eneSteps,errorTargets=errorTargets,choleskyThreshold=choleskyThreshold,orthoSteps=orthoSteps,stochasticIter=stochasticIter,ndets=ndets,nroot=nroot,left=left,right=right)

def run_fpafqmc_mf(mf,norb_frozen=0,mo_coeff=None,chol_cut=1e-5,vmc_root=None,ene0Guess=None,mpi_prefix=None,nproc=None,seed = None,dt = 0.05,nsteps=60,eneSteps= [ 20, 40, 50, 60 ],errorTargets=[ 0.8e-3, 0.8e-3, 0.8e-3, 0.8e-3 ],choleskyThreshold=0.5e-4,orthoSteps=5,stochasticIter=100000,left=None,right=None,ndets=None,ciThreshold=None):
  if(ene0Guess is None): print("Guess energy not given. Using mean field energy");ene0Guess=mf.energy_tot()
  print("\nPreparing AFQMC calculation")
  if vmc_root is None:
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    vmc_root = dir_path + '/../'
  
  if mpi_prefix is None:
  	mpi_prefix = "mpirun "
  	if nproc is not None:
  		mpi_prefix += f" -np {nproc} "
  afqmc_binary = vmc_root + "/bin/DQMC"
  mol = mf.mol
  if mo_coeff is None:
  	if isinstance(mf, scf.uhf.UHF):
  		mo_coeff = mf.mo_coeff[0]
  	elif isinstance(mf, scf.rhf.RHF):
  		mo_coeff = mf.mo_coeff
  else:
  	raise Exception("Invalid mean field object!")
  
  
  print("Calculating Cholesky integrals")
  h1e, chol, nelec, enuc = QMCUtils.generate_integrals(mol, mf.get_hcore(), mo_coeff, chol_cut)
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

  nbasis = h1e.shape[-1]
  print('Size of the correlation space:')
  print(f'Number of electrons: {nelec}')
  print(f'Number of basis functions: {nbasis}')
  print(f'Number of Cholesky vectors: {chol.shape[0]}\n')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - v0
  chol = chol.reshape((chol.shape[0], -1))

  QMCUtils.write_dqmc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')


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
      QMCUtils.writeMat(q,"rhf.txt")
    else:
      q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
      uhfCoeffs[:, :nbasis] = q
      uhfCoeffs[:, nbasis:] = q
      QMCUtils.writeMat(q,"rhf.txt")
    QMCUtils.writeMat(uhfCoeffs, "uhf.txt")

  elif isinstance(mf, scf.rhf.RHF):
    hf_type = "rhf"
    q, r = np.linalg.qr(mo_coeff[:, norb_frozen:].T.dot(overlap).dot(mf.mo_coeff[:, norb_frozen:]))
    QMCUtils.writeMat(q, "rhf.txt")
  
  if(left is None): left = hf_type
  if(right is None): right = hf_type  #If right ==ccsd
  write_fpafqmc_input(right=right,left=left,ene0Guess=ene0Guess,seed = seed,dt = dt,nsteps=nsteps,eneSteps= eneSteps,errorTargets=errorTargets,choleskyThreshold=choleskyThreshold,orthoSteps=orthoSteps,stochasticIter=stochasticIter)
  command = f"export OMP_NUM_THREADS=1; {mpi_prefix} {afqmc_binary} dqmc.json > dqmc.out"
  os.system(command)

def run_fpafqmc_mc(mc,norb_frozen=0,mo_coeff=None,chol_cut=1e-5,vmc_root=None,ene0Guess=None,mpi_prefix=None,nproc=None,seed = None,dt = 0.05,nsteps=60,eneSteps= [ 20, 40, 50, 60 ],errorTargets=[ 0.8e-3, 0.8e-3, 0.8e-3, 0.8e-3 ],choleskyThreshold=0.5e-4,orthoSteps=5,stochasticIter=100000,left=None,right=None,ndets=None,ciThreshold=None,nroot=0,use_eri=False):
  if(left is None): left='multislater'
  if(ene0Guess is None): print("Guess energy not given. Using energy from mc object");ene0Guess=mc.e_tot()
  print("\nPreparing AFQMC calculation")
  if vmc_root is None:
      path = os.path.abspath(__file__)
      dir_path = os.path.dirname(path)
      vmc_root = dir_path + '/../'
  
  if mpi_prefix is None:
      mpi_prefix = "mpirun "
      if nproc is not None:
          mpi_prefix += f" -np {nproc} "
  afqmc_binary = vmc_root + "/bin/DQMC"
  mol = mc.mol
  mo_coeff = mc.mo_coeff
  # calculate cholesky integrals
  print("Calculating Cholesky integrals")
  h1e, chol, nelec, enuc = QMCUtils.generate_integrals(mol, mc.get_hcore(), mo_coeff, chol_cut)

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

  if use_eri: # generate cholesky in mo basis
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

  print("Finished calculating Cholesky integrals\n")

  print('Size of the correlation space:')
  print(f'Number of electrons: {nelec}')
  print(f'Number of basis functions: {nbasis}')
  print(f'Number of Cholesky vectors: {chol.shape[0]}\n')
  chol = chol.reshape((-1, nbasis, nbasis))
  v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
  h1e_mod = h1e - v0
  chol = chol.reshape((chol.shape[0], -1))

  QMCUtils.write_dqmc(h1e, h1e_mod, chol, sum(nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

  # write mo coefficients
  det_file = 'dets.bin'
  if nroot > 0:
    det_file = f'dets_{nroot}.bin'
  norb_act, state, ndets_all = QMCUtils.read_dets(det_file, 1)
  up = np.argsort(-np.array(list(state.keys())[0][0])) + norb_core
  dn = np.argsort(-np.array(list(state.keys())[0][1])) + norb_core + nbasis
  hf_type = "rhf"
  if list(state.keys())[0][0] == list(state.keys())[0][1]:
    rhfCoeffs = np.eye(nbasis)
    QMCUtils.writeMat(rhfCoeffs[:, np.concatenate((range(norb_core), up, range(norb_core+norb_act, nbasis))).astype(int)], "rhf.txt")
  else:
    hf_type = "uhf"
    uhfCoeffs = np.hstack((np.eye(nbasis), np.eye(nbasis)))
    QMCUtils.writeMat(uhfCoeffs[:, np.concatenate((np.array(range(norb_core)), up, np.array(range(norb_core+norb_act, nbasis+norb_core)), dn, np.array(range(nbasis+norb_core+norb_act, 2*nbasis)))).astype(int)], "uhf.txt")
    rhfCoeffs = np.eye(nbasis)
    QMCUtils.writeMat(rhfCoeffs[:, np.concatenate((range(norb_core), up, range(norb_core+norb_act, nbasis))).astype(int)], "rhf.txt")

  if(right is None or (right.lower()=='uhf' or right.lower() == 'rhf')): right = hf_type
  

  # determine ndets for doing calculations
  ndets_list = [ ]
  if isinstance(ndets, list):
    flag = False
    for i, n in enumerate(ndets):
      if n < ndets_all:
        ndets_list.append(n)
      else:
        flag = True
    if len(ndets_list) == 0:
      ndets_list = [ ndets_all ]
    elif flag == True:
      ndets_list.append(ndets_all)
  elif isinstance(ndets, int):
    if ndets > ndets_all:
      ndets = ndets_all
    ndets_list = [ ndets ]
  else:
    raise Exception('Provide ndets as an int or a list of ints!')

  for i, n in enumerate(ndets_list):
    write_fpafqmc_input(right=right,left=left, numAct = norb_act, numCore = norb_core, ndets=n,detFile=det_file,ene0Guess=ene0Guess,seed = seed,dt = dt,nsteps=nsteps,eneSteps= eneSteps,errorTargets=errorTargets,choleskyThreshold=choleskyThreshold,orthoSteps=orthoSteps,stochasticIter=stochasticIter,ciThreshold=ciThreshold,filename=f"dqmc_{n}.json")
    command = f"export OMP_NUM_THREADS=1; {mpi_prefix} {afqmc_binary} dqmc_{n}.json > dqmc_{n}.out"
    os.system(command)    



