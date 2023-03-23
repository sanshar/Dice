import os
import numpy
import numpy as np
import scipy

import QMCUtils
from functools import reduce

from pyscf import gto, scf, lib
import fcidump_rel
#import x2camf_hf

def generate_integrals(mf, norb_core, ncore_frozen, nvirt_frozen, mf_type='g', chol_cut=1e-5):
    # mf_type can be either g or j, representing mf in spherical ao or j-adapted spinors
    print('generating integrals')
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    chol_vecs = QMCUtils.chunked_cholesky(mol, max_error=chol_cut)
    h1e = reduce(numpy.dot, (mo_coeff.T.conj(), mf.get_hcore(), mo_coeff))

    # ncore_frozen : not correlated in afqmc, only contribute to 1-body potential
    # nvirt_frozen: freeze virtual orbital from top.
    # norb_core : correlated in afqmc but not in trial hci state

    nbasis = h1e.shape[-1]
    nelec = sum(mol.nelec)
    nelec_afqmc = nelec - ncore_frozen
    print(f"nbasis:{nbasis},\nnelec:{nelec},\n{nelec_afqmc}")
    assert (ncore_frozen < nelec)
    mo_coeff = mo_coeff[:, ncore_frozen:nbasis - nvirt_frozen]
    hcore = mf.get_hcore(mf.mol)
    core_occ = numpy.zeros(len(mf.mo_energy))
    core_occ[:ncore_frozen] = 1.0
    core_dm = mf.make_rdm1(mo_occ=core_occ)
    corevhf = mf.get_veff(mol, core_dm)
    energy_core = mf.energy_nuc()
    energy_core += numpy.einsum('ij,ji', core_dm, hcore)
    energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
    h1eff = reduce(numpy.dot, (mo_coeff.T.conj(), hcore + corevhf, mo_coeff))
    nmo = h1eff.shape[-1]
    nchol = chol_vecs.shape[0]
    chol = numpy.zeros((nchol, nmo, nmo), dtype=complex)
    c = mol.sph2spinor_coeff()
    c2 = numpy.vstack(c)
    for i in range(nchol):
        chol_i = chol_vecs[i].reshape(nbasis//2, nbasis//2)
        if mf_type is 'g':
            chol_i = scipy.linalg.block_diag(chol_i, chol_i)
            chol[i] = reduce(numpy.dot, (mo_coeff.T.conj(), chol_i, mo_coeff))
        elif mf_type is 'j':
            chol_i = scipy.linalg.block_diag(chol_i, chol_i)
            chol_i = lib.einsum('ip,pq,qj->ij',c2.T.conj(), chol_i, c2)
            chol[i] = lib.einsum('ip,pq,qj',mo_coeff.T.conj(), chol_i, mo_coeff)
    print(f'nelec:{nelec-ncore_frozen}')
    print(f'nmo:{nmo}')
    print(f'chol.shape{chol.shape}')
    chol.reshape((-1, nmo, nmo))
    print(chol.shape)
    v0 = 0.5*numpy.einsum('nik,nkj->ij', chol, chol, optimize='optimal')
    h1e_mod = h1eff - v0
    chol = chol.reshape((chol.shape[0], -1))
    #chol=chol[0:2]
    QMCUtils.write_dqmc_gzhf(h1eff, h1e_mod, chol, nelec_afqmc, nmo, energy_core.real)

def run_afqmc_gzhf_multislater(
    mf, norb_core, ncore_frozen, nvirt_frozen,
    dqmc_binary='DQMC',
    integral=None, left_wave='multislater',
    vmc_root=None, mpi_prefix=None, nproc=None,
    chol_cut=1e-5, ndets=100, iroot=0, seed=None,
    dt=0.005, steps_per_block=50, nwalk_per_proc=5,
    nblocks=1000, ortho_steps=20, burn_in=50,
    cholesky_threshold=1.0e-3, weight_cap=None,
    write_one_rdm=False, run_dir=None,
    phaseless_error=None, scratch_dir=None):

    owd = os.getcwd()
    if integral is None:
        pass
    else:
        integral = os.path.abspath(integral)

    det_file = 'dets.bin'
    if iroot > 0:
        det_file = f'dets_{iroot}.bin'

    if run_dir is not None:
        if os.path.isdir(run_dir):
            pass
        else:
            os.system(f"mkdir -p {run_dir};")
        if os.path.isfile(det_file):
            os.system(f"cp {det_file} {run_dir}")
        else:
            print(f'{det_file} does not exist, use ghf determinant instead')
        os.chdir(f'{run_dir}')
        if scratch_dir is not None:
            os.system(f"mkdir -p {scratch_dir}")

    if integral is None:
        print('generating integrals')
        integral='FCIDUMP_chol'
        mol = mf.mol
        mo_coeff = mf.mo_coeff
        chol_vecs = QMCUtils.chunked_cholesky(mol, max_error=chol_cut)
        h1e = reduce(numpy.dot, (mo_coeff.T.conj(), mf.get_hcore(), mo_coeff))

        # ncore_frozen : not correlated in afqmc, only contribute to 1-body potential
        # nvirt_frozen: freeze virtual orbital from top.
        # norb_core : correlated in afqmc but not in trial hci state
        nbasis = h1e.shape[-1]
        nelec = sum(mol.nelec)
        nelec_afqmc = nelec - ncore_frozen
        print(f"nbasis:{nbasis},\nnelec:{nelec},\n{nelec_afqmc}")
        assert (ncore_frozen < nelec)
        mo_coeff = mo_coeff[:, ncore_frozen:nbasis - nvirt_frozen]
        hcore = mf.get_hcore(mf.mol)
        core_occ = numpy.zeros(len(mf.mo_energy))
        core_occ[:ncore_frozen] = 1.0
        core_dm = mf.make_rdm1(mo_occ=core_occ)
        corevhf = mf.get_veff(mol, core_dm)
        energy_core = mf.energy_nuc()
        energy_core += numpy.einsum('ij,ji', core_dm, hcore)
        energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
        h1eff = reduce(numpy.dot, (mo_coeff.T.conj(), hcore + corevhf, mo_coeff))
        nmo = h1eff.shape[-1]
        nchol = chol_vecs.shape[0]
        chol = numpy.zeros((nchol, nmo, nmo), dtype=complex)
        for i in range(nchol):
            chol_i = chol_vecs[i].reshape(nbasis//2, nbasis//2)
            chol_i = scipy.linalg.block_diag(chol_i, chol_i)
            chol[i] = reduce(numpy.dot, (mo_coeff.T.conj(), chol_i, mo_coeff))

        print(f'nelec:{nelec-ncore_frozen}')
        print(f'nmo:{nmo}')
        print(f'chol.shape{chol.shape}')
        chol.reshape((-1, nmo, nmo))
        print(chol.shape)
        v0 = 0.5*numpy.einsum('nik,nkj->ij', chol, chol, optimize='optimal')
        h1e_mod = h1eff - v0
        chol = chol.reshape((chol.shape[0], -1))
        #chol=chol[0:2]
        QMCUtils.write_dqmc_gzhf(h1eff, h1e_mod, chol, nelec_afqmc, nmo, energy_core.real)
        #QMCUtils.prepAFQMC_gzhf(mf)
        # write mo coefficients

    nmo = mf.mol.nao_2c()-ncore_frozen-nvirt_frozen
    print(ncore_frozen, nvirt_frozen)
    print(nmo)
    ghfCoeffs = np.eye(nmo)

    ndets_list = []
    det_file = 'dets.bin'
    if iroot > 0:
        det_file = f'dets_{iroot}.bin'

    if os.path.isfile(det_file):
        norb_act, state, ndets_all = QMCUtils.read_dets_gzhf(det_file, 1)
        print(mf.mol.nao_2c(), ncore_frozen, nvirt_frozen)
        masked_det = -np.array(list(state.keys())[0]) + 0.0*np.arange(len(list(state.keys())[0])) + norb_core
        print(masked_det)
        occ_first = np.argsort(masked_det)
        det_order = np.concatenate((range(norb_core), 
                                    occ_first+norb_core, 
                                    range(norb_core+norb_act, nmo))).astype(int)
        print(det_order)
        ghfCoeffs = ghfCoeffs[:,det_order]
        if isinstance(ndets, int):
            if ndets > ndets_all:
                ndets = ndets_all
            ndets_list = [ndets]
        else:
            raise Exception('Provide ndets as an int!')
        norb_act = norb_act // 2
        norb_core = norb_core // 2
    else:
        ndets_list=[ndets]
        norb_act = None
        norb_core = None

    QMCUtils.writeMat(ghfCoeffs, "ghf.txt")

    if mpi_prefix is None:
        mpi_prefix = 'mpirun '
    if nproc is not None:
        mpi_prefix += f'-np {nproc}'
    os.system('rm -f samples.dat')
    print(ndets_list)
    e_afqmc = [None for _ in ndets_list]
    err_afqmc = [None for _ in ndets_list]

    for i, n in enumerate(ndets_list):
        QMCUtils.write_afqmc_input(seed=seed,
                                   integral=integral,
                                   numAct=norb_act,
                                   numCore=norb_core,
                                   intType='gz',
                                   left=left_wave,
                                   right='ghf',
                                   ndets=n,
                                   detFile=det_file,
                                   dt=dt,
                                   nsteps=steps_per_block,
                                   nwalk=nwalk_per_proc,
                                   stochasticIter=nblocks,
                                   orthoSteps=ortho_steps,
                                   burnIter=burn_in,
                                   choleskyThreshold=cholesky_threshold,
                                   weightCap=weight_cap,
                                   writeOneRDM=write_one_rdm,
                                   phaselessErrorTarget=phaseless_error,
                                   fname=f'afqmc_{n}.json')
        print(f'Stating AFQMC / HCI ({n} dets) calculation', flush=True)
        command = f'export OMP_NUM_THREADS=1; {mpi_prefix} {dqmc_binary} afqmc_{n}.json > afqmc.log'
        os.system(command)

        if (os.path.isfile('samples.dat')):
            print('\nBlocking analysis:', flush=True)
            command = f"mv samples.dat samples_{n}.dat; mv afqmc.dat afqmc_{n}.dat; rm dets*.bin;"\
                      f"mv blocking.tmp blocking_{n}.out; "\
                      f"cat blocking_{n}.out"
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
                print(f'AFQMC energy: {e_afqmc[i]}\n', flush=True)
                print('Could not find a stochastic error estimate, check blocking analysis\n', flush=True)
        else:
            print("\nAFQMC calculation did not finish, check the afqmc.dat file\n")
            exit(1)
    os.chdir(owd)

    return e_afqmc, err_afqmc
