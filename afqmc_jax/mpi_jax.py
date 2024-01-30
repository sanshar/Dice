import ph_afqmc, ph_afqmc_MultiSlater
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import h5py, json, QMCUtils
with h5py.File('FCIDUMP_chol', 'r') as fh5:
  [nelec, nmo, ms, nchol] = fh5['header']
  h0 = np.array(fh5.get('energy_core'))
  h1 = np.array(fh5.get('hcore')).reshape(nmo, nmo)
  chol = np.array(fh5.get('chol')).reshape(-1, nmo, nmo)

with open('afqmc.json') as fjson:
  d = json.load(fjson)
print("INPUT FIKLE:")
print(json.dumps(d, indent=2))

  
numCore = d['system'].get('numCore',0)

if d['wavefunction']['left'] == 'multislater':
  Acre, Ades, Bcre, Bdes, coeff = \
    QMCUtils.getExcitation(numCore, d['wavefunction']['determinants'], d['wavefunction']['ndets'], \
      maxExcitation=ph_afqmc_MultiSlater.MAX_EXCITATION)

excitations = [ Acre, Ades, Bcre, Bdes, coeff]
  
norb = nmo
nelec = nelec // 2

dt = d['sampling']['dt']
nwalkers = d['sampling']['nwalk']
nsteps = d['sampling']['nsteps']
nblocks = d['sampling']['stochasticIter']
seed = d['sampling']['seed']
nclub = 1 #d['sampling']['orthoSteps']

#dt = 0.01
#nwalkers = 50
#nsteps = 50
#nblocks = 5
#seed = np.random.randint(1, 1e6)
#nclub = 200

import time
init = time.time()
comm.Barrier()
#ph_afqmc.run_afqmc(h0, h1, chol, nelec, dt, nwalkers, nsteps, nblocks, seed=seed, neql=1, rdmQ=False, nclub=nclub)
ph_afqmc_MultiSlater.run_afqmc(h0, h1, chol, nelec, dt, nwalkers, nsteps, excitations, nblocks, seed=seed, neql=1, rdmQ=False, nclub=nclub)
comm.Barrier()
end = time.time()
if rank == 0:
  print(f'ph_afqmc_jax walltime: {end - init}', flush=True)

comm.Barrier()
