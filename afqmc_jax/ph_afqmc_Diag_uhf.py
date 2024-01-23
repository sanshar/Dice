import os, time
import numpy as np
from numpy.random import Generator, MT19937, PCG64
import scipy as sp
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, jit, vmap, random, vjp, checkpoint
from mpi4py import MPI

from functools import partial
print = partial(print, flush=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
  print(f'Number of cores: {size}\n', flush=True)


def blocking_analysis(weights, energies, neql, printQ=False):
  nSamples = weights.shape[0] - neql
  weights = weights[neql:]
  energies = energies[neql:]
  weightedEnergies = np.multiply(weights, energies)
  meanEnergy = weightedEnergies.sum() / weights.sum()
  if printQ:
    print(f'\nMean energy: {meanEnergy:.8e}')
    print('Block size    # of blocks        Mean                Error')
  blockSizes = np.array([ 1, 2, 5, 10, 20, 50, 70, 100, 200, 500 ])
  prevError = 0.
  plateauError = None
  for i in blockSizes[blockSizes < nSamples/2.]:
    nBlocks = nSamples//i
    blockedWeights = np.zeros(nBlocks)
    blockedEnergies = np.zeros(nBlocks)
    for j in range(nBlocks):
      blockedWeights[j] = weights[j*i:(j+1)*i].sum()
      blockedEnergies[j] = weightedEnergies[j*i:(j+1)*i].sum() / blockedWeights[j]
    v1 = blockedWeights.sum()
    v2 = (blockedWeights**2).sum()
    mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
    error = (np.multiply(blockedWeights, (blockedEnergies - mean)**2).sum() / (v1 - v2 / v1) / (nBlocks - 1))**0.5
    if printQ:
      print(f'  {i:4d}           {nBlocks:4d}       {mean:.8e}       {error:.6e}')
    if error < 1.05 * prevError and plateauError is None:
      plateauError = max(error, prevError)
    prevError = error

  if printQ:
    if plateauError is not None:
      print(f'Stocahstic error estimate: {plateauError:.6e}\n')

  return meanEnergy, plateauError


@jit
def calc_overlap(trialA, trialB, walkerA, walkerB):
  return jnp.linalg.det(trialA.dot(walkerA))* jnp.linalg.det(trialB.dot(walkerB))

calc_overlap_vmap = vmap(calc_overlap, in_axes=(None,None, 0,0))


@jit
def calc_green(trial, walker):
  Sinv = jnp.linalg.inv(trial.conj().dot(walker))
  return walker.dot( jnp.dot(Sinv, trial.conj())).T

calc_green_vmap = vmap(calc_green, in_axes = (None,0) )


@jit
def calc_force_bias(trialA, walkerA, trialB, walkerB, vdiagChol):
  Sinv = jnp.linalg.inv(trialA.dot(walkerA))
  G1 = jnp.dot(Sinv, trialA)
  Green = jnp.einsum('xi,ix->x', walkerA, G1)

  Sinv = jnp.linalg.inv(trialB.dot(walkerB))
  G1 = jnp.dot(Sinv, trialB)
  Green += jnp.einsum('xi,ix->x', walkerB, G1)

  fb = jnp.dot(Green, vdiagChol)

  return fb

calc_force_bias_vmap = vmap(calc_force_bias, in_axes = (None, 0, None, 0, None))


@jit
def calc_energy(h0, h1, vdiag, trialA, trialB, walkerA, walkerB):
  ene0 = h0
  green_walkerA, green_walkerB = calc_green(trialA, walkerA), calc_green(trialB, walkerB)
  ene1 = jnp.sum((green_walkerA+green_walkerB) * h1)

  Gdiag = (green_walkerA + green_walkerB).diagonal()
  J = jnp.dot(Gdiag, jnp.dot(vdiag, Gdiag))/2.

  KA = jnp.einsum('ij,ji,ij', green_walkerA, vdiag, green_walkerA.T)
  KB = jnp.einsum('ij,ji,ij', green_walkerB, vdiag, green_walkerB.T)
  #K = jnp.sum(green_walker*green_walker*vdiag)
  return J - (KA + KB)/2. + ene1 + ene0

calc_energy_vmap = vmap(calc_energy, in_axes = (None, None, None, None, None, 0, 0))


# defining this separately because calculating vhs for a batch seems to be faster
@jit
def apply_propagator(exp_h1, vhs_i, walker_iA, walker_iB):
  walker_iA = exp_h1.dot(walker_iA)
  walker_iB = exp_h1.dot(walker_iB)
  #walker_i = jnp.dot(Uortho.T, jnp.dot(Uortho, walker_i))
  
  walker_iA = jnp.einsum('x,xi->xi', vhs_i, walker_iA)
  walker_iB = jnp.einsum('x,xi->xi', vhs_i, walker_iB)

  walker_iA = exp_h1.dot(walker_iA)
  walker_iB = exp_h1.dot(walker_iB)
  #walker_i = jnp.dot(Uortho.T, jnp.dot(Uortho, walker_i))
  return walker_iA, walker_iB

@jit
def apply_propagator_vmap(exp_h1, vdiagChol, dt, walkersA, walkersB, fields):
  vhs = jnp.exp(1.j * jnp.sqrt(dt) * fields.dot(vdiagChol.T)).reshape(walkersA.shape[0], walkersA.shape[1])
  #vhs = 1.j * jnp.sqrt(dt) * fields.dot(chol).reshape(walkers.shape[0], walkers.shape[1], walkers.shape[1])
  return vmap(apply_propagator, in_axes = (None, 0, 0, 0))(exp_h1, vhs, walkersA, walkersB )


# one block of phaseless propagation consisting of nsteps steps followed by energy evaluation
def propagate_phaseless(h0_prop, h0, h1, vdiag, vdiagChol, trialA, trialB, dt, walkersA, walkersB, weights, e_estimate, mf_shifts, fields_np, zeta_sr):

  ##I don't think there is an exchange term
  #v0 = 0.5 * jnp.einsum('gik,gjk->ij', chol.reshape(-1, h1.shape[0], h1.shape[0]), chol.reshape(-1, h1.shape[0], h1.shape[0]), optimize='optimal')
  v0 = 0.5 * jnp.einsum('ig,ig->i', vdiagChol, vdiagChol, optimize='optimal')
  h1_mod = 1.*h1 - jnp.diag(v0)

  h1_mod = h1_mod - jnp.diag(1.j*mf_shifts.dot(vdiagChol.T))
  exp_h1 = jsp.linalg.expm(-dt * h1_mod / 2.)

  #force_bias = calc_force_bias_vmap(trialA, walkersA, trialB, walkersB, vdiagChol)
  #print(force_bias)
  #print(mf_shifts)
  #exit(0)
  #field_shifts = -jnp.sqrt(dt) * (1.j * force_bias - mf_shifts)
  #print(fields_np.shape, field_shifts.shape)
  #shifted_fields = fields_np - field_shifts
  #print(mf_shifts)
  #walker2 = apply_propagator_vmap(exp_h1, vdiagChol, dt, walkers, shifted_fields)

  #print(trial.shape, walker2.shape)
  #print(calc_overlap(trial, walker2))
  #print(calc_energy(h0, h1, vdiag, trial, walker2[0]))
  #exit(0)


  #w = jnp.linalg.qr(walkers[0])[0]
  #print(trial.shape, w.shape)
  #print(calc_overlap(trial, w))
  
  # carry : [ walkers, weights, overlaps, e_shift ]
  #key = random.PRNGKey(seed)
  @checkpoint
  def scanned_fun(carry, x):
    fields = jnp.array(x)
    force_bias = calc_force_bias_vmap(trialA, carry[0], trialB, carry[1], vdiagChol)
    field_shifts = -jnp.sqrt(dt) * (1.j * force_bias - mf_shifts)
    #print(fields.shape, force_bias.shape, field_shifts.shape)
    shifted_fields = fields - field_shifts
    shift_term = jnp.sum(shifted_fields * mf_shifts, axis=1)
    fb_term = jnp.sum(fields * field_shifts - field_shifts * field_shifts / 2., axis=1)
    carry[0] = qr_vmap(carry[0])
    carry[1] = qr_vmap(carry[1])
    carry[3] = calc_overlap_vmap(trialA, trialB, carry[0], carry[1])
    carry[0], carry[1] = apply_propagator_vmap(exp_h1, vdiagChol, dt, carry[0], carry[1], shifted_fields)

    overlaps_new = calc_overlap_vmap(trialA, trialB, carry[0], carry[1])
    imp_fun = jnp.exp(-jnp.sqrt(dt) * shift_term + fb_term + dt * (carry[4] + h0_prop)) * overlaps_new / carry[3]
    theta = jnp.angle(jnp.exp(-jnp.sqrt(dt) * shift_term) * overlaps_new / carry[3])
    imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
    imp_fun_phaseless = jnp.where(imp_fun_phaseless < 1.e-3, 0., imp_fun_phaseless)
    imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100., 0., imp_fun_phaseless)
    imp_fun_phaseless = jnp.where(jnp.isnan(imp_fun_phaseless), 0., imp_fun_phaseless)
    carry[2] = imp_fun_phaseless * carry[2]
    carry[2] = jnp.where(carry[2] > 100., 0., carry[2])
    carry[3] = overlaps_new
    carry[4] = e_estimate - 0.1 * jnp.log(jnp.sum(carry[2]) / carry[0].shape[0]) / dt
    return carry, x


  # carry : [ walkers, weights, overlaps, e_shift, block_energy ]
  def outer_scanned_fun(carry, x):
    carry[:5], _ = lax.scan(scanned_fun, carry[:5], x[0])
    carry[3] = calc_overlap_vmap(trialA, trialB, carry[0], carry[1])
    energy_samples = jnp.real(calc_energy_vmap(h0, h1, vdiag, trialA, trialB, carry[0], carry[1]))
    energy_samples = jnp.where(jnp.abs(energy_samples - carry[4]) > jnp.sqrt(2./dt), carry[4], energy_samples)
    block_energy = jnp.sum(energy_samples * carry[2]) / jnp.sum(carry[2])
    carry[5] = block_energy
    carry[4] = 0.9 * carry[4] + 0.1 * block_energy
    carry[0], carry[1], carry[2] = stochastic_reconfiguration(carry[0], carry[1], carry[2], x[1])
    return carry, x

  block_energy = 0
  overlaps = calc_overlap_vmap(trialA, trialB, walkersA, walkersB)


  
  [ walkersA, walkersB, weights, overlaps, e_shift, block_energy ], _ = lax.scan(outer_scanned_fun, [ walkersA, walkersB, weights, overlaps, e_estimate, block_energy ], (fields_np, zeta_sr))
  #print(rank, block_energy)
  return block_energy, (walkersA, walkersB, weights)

propagate_phaseless_jit = jit(propagate_phaseless)


# called once after each block
def qr_vmap(walkers):
  walkers, _ = vmap(jnp.linalg.qr)(walkers)
  return walkers

norm_vmap = vmap(jnp.linalg.norm, in_axes=1)

def normalize(walker):
  return walker / norm_vmap(walker)

normalize_vmap = vmap(normalize)


# this uses numpy but is only called once after each block
def stochastic_reconfiguration_np(walkers, weights, zeta):
  nwalkers = walkers.shape[0]
  walkers = np.array(walkers)
  weights = np.array(weights)
  walkers_new = 0. * walkers
  cumulative_weights = np.cumsum(np.abs(weights))
  total_weight = cumulative_weights[-1]
  average_weight = total_weight / nwalkers
  weights_new = np.ones(nwalkers) * average_weight
  for i in range(nwalkers):
    z = (i + zeta) / nwalkers
    new_i = np.searchsorted(cumulative_weights, z * total_weight)
    walkers_new[i] = walkers[new_i].copy()
  return jnp.array(walkers_new), jnp.array(weights_new)


@jit
def stochastic_reconfiguration(walkersA, walkersB, weights, zeta):
  nwalkers = walkersA.shape[0]
  cumulative_weights = jnp.cumsum(jnp.abs(weights))
  total_weight = cumulative_weights[-1]
  average_weight = total_weight / nwalkers
  weights = jnp.ones(nwalkers) * average_weight
  z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
  indices = vmap(jnp.searchsorted, in_axes = (None, 0))(cumulative_weights, z)
  walkersA, walkersB = walkersA[indices], walkersB[indices]
  return walkersA, walkersB, weights


# this uses numpy but is only called once after each block
def stochastic_reconfiguration_mpi(walkersA, walkersB, weights, zeta):
  nwalkers = walkersA.shape[0]
  walkersA = np.array(walkersA)
  walkersB = np.array(walkersB)

  weights = np.array(weights)
  walkersA_new = 0. * walkersA
  walkersB_new = 0. * walkersB
  weights_new = 0. * weights

  global_buffer_walkersA = None
  global_buffer_walkersA_new = None
  global_buffer_walkersB = None
  global_buffer_walkersB_new = None
  global_buffer_weights = None
  global_buffer_weights_new = None
  if rank == 0:
    global_buffer_walkersA = np.zeros((nwalkers * size, walkersA.shape[1], walkersA.shape[2]), dtype=walkersA.dtype)
    global_buffer_walkersA_new = np.zeros((nwalkers * size, walkersA.shape[1], walkersA.shape[2]), dtype=walkersA.dtype)

    global_buffer_walkersB = np.zeros((nwalkers * size, walkersB.shape[1], walkersB.shape[2]), dtype=walkersB.dtype)
    global_buffer_walkersB_new = np.zeros((nwalkers * size, walkersB.shape[1], walkersB.shape[2]), dtype=walkersB.dtype)

    global_buffer_weights = np.zeros(nwalkers * size, dtype=weights.dtype)
    global_buffer_weights_new = np.zeros(nwalkers * size, dtype=weights.dtype)

  comm.Gather(walkersA, global_buffer_walkersA, root=0)
  comm.Gather(walkersB, global_buffer_walkersB, root=0)
  comm.Gather(weights, global_buffer_weights, root=0)

  if rank == 0:
    cumulative_weights = np.cumsum(np.abs(global_buffer_weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers / size
    global_buffer_weights_new = (np.ones(nwalkers * size) * average_weight).astype(weights.dtype)
    for i in range(nwalkers * size):
      z = (i + zeta) / nwalkers / size
      new_i = np.searchsorted(cumulative_weights, z * total_weight)
      global_buffer_walkersA_new[i] = global_buffer_walkersA[new_i].copy()
      global_buffer_walkersB_new[i] = global_buffer_walkersB[new_i].copy()

  comm.Scatter(global_buffer_walkersA_new, walkersA_new, root=0)
  comm.Scatter(global_buffer_walkersB_new, walkersB_new, root=0)
  comm.Scatter(global_buffer_weights_new, weights_new, root=0)
  return jnp.array(walkersA_new), jnp.array(walkersB_new), jnp.array(weights_new)


def run_afqmc(h0, h1, PsiT, Psi0, vdiag, vdiagChol, nup, ndn, dt, nwalkers, nsteps, nblocks, neql=50, seed=0, rdmQ=False, nclub=1):
  
  init = time.time()
  norb = h1.shape[0]
  nchol = vdiagChol.shape[1]
  trialA, trialB = (1+0.j)*PsiT[0][:nup,:], (1+0.j)*PsiT[1][:ndn,:]

  nocc = np.einsum('ix,ix->x', trialA, trialA) + np.einsum('ix,ix->x', trialB, trialB)
  mf_shifts = 1.j * np.dot(nocc, vdiagChol)

  h0_prop = - h0 - np.sum(mf_shifts**2) / 2.

  mf_shifts = jnp.array(mf_shifts)
  h0_prop = jnp.array(h0_prop)
  h0 = jnp.array(h0)
  h1 = jnp.array(h1)
  trialA, trialB = jnp.array(trialA), jnp.array(trialB)
  vdiag = jnp.array(vdiag)
  vdiagChol = jnp.array(vdiagChol)
  
  global_block_weights = np.zeros(nblocks)
  local_block_weights = np.zeros(nblocks)
  weights = jnp.ones(nwalkers)
  global_block_weights[0] = nwalkers * size
  local_block_weights[0] = nwalkers
  global_block_energies = np.zeros(nblocks)
  #local_block_rdm1 = np.zeros((nblocks, norb, norb))

  walkersA = jnp.stack([(1.+0.j)*Psi0[0].T for _ in range(nwalkers)])
  walkersB = jnp.stack([(1.+0.j)*Psi0[1].T for _ in range(nwalkers)])

  energy_samples = jnp.real(calc_energy_vmap(h0, h1, vdiag, trialA, trialB, walkersA, walkersB))
  global_block_energies[0] = jnp.sum(energy_samples) / nwalkers   # assuming identical walkers
  e_estimate = jnp.array(global_block_energies[0])
  total_block_energy_n = np.zeros(1, dtype='float64')
  total_block_weight_n = np.zeros(1, dtype='float64')

  rng = Generator(MT19937(seed + rank))
  #rng = Generator(MT19937(seed))
  #hf_rdm = 2 * np.eye(norb, nelec).dot(np.eye(norb, nelec).T)

  #print("walkers shape ", walkers.shape)
  comm.Barrier()
  init_time = time.time() - init
  if rank == 0:
    print("   Iter        Mean energy          Stochastic error       Walltime")
    n = 0
    print(f" {n:5d}      {global_block_energies[0]:.9e}                -              {init_time:.2e} ")
  comm.Barrier()


  force_bias = calc_force_bias_vmap(trialA, walkersA, trialB, walkersB, vdiagChol)
  #print(force_bias, nchol)
  for n in range(1, nblocks):
    fields_np = rng.standard_normal(size=(nclub, nsteps, walkersA.shape[0], nchol))
    zeta_sr = rng.uniform(size=(nclub,))

    if rdmQ:
      block_energy_n, block_vjp, (walkers, weights) = vjp(propagate_phaseless_jit, h0_prop, h0, h1, chol, rot_chol, dt, walkers, weights, e_estimate, mf_shifts, fields_np, zeta_sr, has_aux=True)
      local_block_rdm1[n] = np.array(block_vjp(1.)[2])
      if np.isnan(np.sum(local_block_rdm1[n])) or np.linalg.norm(local_block_rdm1[n] - hf_rdm) > 1.e4:
        local_block_rdm1[n] = np.zeros((norb, norb))
    else:
      block_energy_n, (walkersA, walkersB, weights) = propagate_phaseless(h0_prop, h0, h1, vdiag, vdiagChol, trialA, trialB, dt, walkersA, walkersB, weights, e_estimate, mf_shifts, fields_np, zeta_sr)

    local_block_weights[n] = jnp.sum(weights)
    block_energy_n = np.array([block_energy_n], dtype='float64')
    block_weight_n = np.array([jnp.sum(weights)], dtype='float64')
    block_weighted_energy_n = np.array([block_energy_n * block_weight_n], dtype='float64')

    comm.Reduce([block_weighted_energy_n, MPI.DOUBLE], [total_block_energy_n, MPI.DOUBLE], op=MPI.SUM, root=0)
    comm.Reduce([block_weight_n, MPI.DOUBLE], [total_block_weight_n, MPI.DOUBLE], op=MPI.SUM, root=0)
    if rank == 0:
      block_weight_n = total_block_weight_n
      block_energy_n = total_block_energy_n / total_block_weight_n
    comm.Bcast(block_weight_n, root=0)
    comm.Bcast(block_energy_n, root=0)
    global_block_weights[n] = block_weight_n
    global_block_energies[n] = block_energy_n
    walkersA = qr_vmap(walkersA)
    walkersB = qr_vmap(walkersB)
    zeta = rng.uniform()

    walkersA, walkersB, weights = stochastic_reconfiguration_mpi(walkersA, walkersB, weights, zeta)
    #walkers, weights = stochastic_reconfiguration_np(walkers, weights, zeta)
    e_estimate = 0.9 * e_estimate + 0.1 * global_block_energies[n];
    #print(n, block_energy_n)
    
    if n > neql and n%(10) == 0:
      comm.Barrier()
      if rank == 0:
        e_afqmc, energy_error = blocking_analysis(global_block_weights[:n], global_block_energies[:n], neql=neql)
        if energy_error is not None:
          print(f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {time.time() - init:.2e} ", flush=True)
        else:
          print(f" {n:5d}      {e_afqmc:.9e}                -              {time.time() - init:.2e} ", flush=True)
        np.savetxt('samples_jax.dat', np.stack((global_block_weights[:n], global_block_energies[:n])).T)
      comm.Barrier()

  if rdmQ:
    local_weighted_rdm1 = np.array(np.einsum('bij,b->ij', local_block_rdm1[neql:], local_block_weights[neql:]), dtype='float64')
    global_weighted_rdm1 = 0. * local_weighted_rdm1
    global_total_weight = np.zeros(1, dtype='float64')
    local_total_weight = np.array([np.sum(local_block_weights[neql:])], dtype='float64')
    comm.Reduce([local_weighted_rdm1, MPI.FLOAT], [global_weighted_rdm1, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([local_total_weight, MPI.FLOAT], [global_total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Barrier()
    with open(f'rdm_{rank}.dat', "w") as fh:
      fh.write(f'{local_total_weight[0]}\n')
      np.savetxt(fh, local_weighted_rdm1 / local_total_weight)
    if rank == 0:
      rdm1 = global_weighted_rdm1 / global_total_weight
      np.savetxt('rdm1_afqmc.txt', rdm1)
      print(f'\nrdm1_tr: {np.trace(rdm1)}', flush=True)

  comm.Barrier()
  if rank == 0:
    np.savetxt('samples_jax.dat', np.stack((global_block_weights, global_block_energies)).T)
    e_afqmc, err_afqmc = blocking_analysis(global_block_weights, global_block_energies, neql=neql, printQ=True)
    if err_afqmc is not None:
      sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
      sig_err = np.around(np.round(err_afqmc * 10**sig_dec) * 10**(-sig_dec), sig_dec)
      sig_e = np.around(e_afqmc, sig_dec)
      print(f'AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n')
    elif e_afqmc is not None:
      print(f'AFQMC energy: {e_afqmc}\nCould not find a stochastic error estimate, check blocking analysis\n', flush=True)
  comm.Barrier()