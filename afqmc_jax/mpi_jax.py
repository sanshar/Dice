import ph_afqmc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import h5py
with h5py.File('FCIDUMP_chol', 'r') as fh5:
  [nelec, nmo, ms, nchol] = fh5['header']
  h0 = np.array(fh5.get('energy_core'))
  h1 = np.array(fh5.get('hcore')).reshape(nmo, nmo)
  chol = np.array(fh5.get('chol')).reshape(-1, nmo, nmo)

norb = nmo
nelec = nelec // 2

#dt = 0.01
#nwalkers = 100
#nsteps = 800
#nblocks = 500
#seed = np.random.randint(1, 1e6)

dt = 0.01
nwalkers = 100
nsteps = 50
nblocks = 400
seed = np.random.randint(1, 1e6)

import time
init = time.time()
comm.Barrier()
ph_afqmc.run_afqmc(h0, h1, chol, nelec, dt, nwalkers, nsteps, nblocks, seed=seed, neql=10, rdmQ=True, nclub=15)
comm.Barrier()
end = time.time()
if rank == 0:
  print(f'ph_afqmc_jax walltime: {end - init}', flush=True)

comm.Barrier()
