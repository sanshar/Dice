#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Sandeep Sharma <sanshar@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
Internal-contracted MPS perturbation method.  You can contact Sandeep
Sharma for the "icpt" program required by this module.  If this method
is used in your work, please cite
S. Sharma and G. Chan,  J. Chem. Phys., 136 (2012), 124121
S. Sharma, G. Jeanmairet, and A. Alavi,  J. Chem. Phys., 144 (2016), 034103
'''

import pyscf
import os
import time
import tempfile
from functools import reduce
import numpy as np
import h5py
from pyscf import lib
from pyscf.lib import logger
from pyscf import mcscf
from pyscf import ao2mo
from pyscf import scf
from pyscf.ao2mo import _ao2mo
from pyscf import tools
import sys

#in state average calculationg E1eff will be different than E1
#this means that the h1eff in the fock operator which is stored in eris_sp['h1eff'] will be
#calculated using the E1eff and will in general not result in diagonal matrices
def writeNEVPTIntegrals(mc, E1, E2, E1eff, nfro, intfolder):
    # Initializations
    ncor = mc.ncore
    nact = mc.ncas
    norb = mc.mo_coeff.shape[1]
    nvir = norb-ncor-nact
    nocc = ncor+nact
    mo   = mc.mo_coeff




    # (Note: Integrals are in chemistry notation)
    start = time.time()
    print('Producing the integrals')
    eris = _ERIS(mc, mo)
    eris_sp={}

    # h1eff
    eris_sp['h1eff']= eris['h1eff']
    eris_sp['h1eff'][:ncor,:ncor] += np.einsum('abcd,cd', eris['ppaa'][:ncor,:ncor,:,:], E1eff)
    eris_sp['h1eff'][:ncor,:ncor] -= np.einsum('abcd,bd', eris['papa'][:ncor,:,:ncor,:], E1eff)*0.5
    eris_sp['h1eff'][nocc:,nocc:] += np.einsum('abcd,cd', eris['ppaa'][nocc:,nocc:,:,:], E1eff)
    eris_sp['h1eff'][nocc:,nocc:] -= np.einsum('abcd,bd', eris['papa'][nocc:,:,nocc:,:], E1eff)*0.5
    np.save(intfolder+"int1eff",np.asfortranarray(eris_sp['h1eff'][nfro:,nfro:]))
    np.save(intfolder+"f",np.asfortranarray(eris_sp['h1eff'][nfro:,nfro:])) 
    # CVCV
    eriscvcv = eris['cvcv']
    if (not isinstance(eris_sp['h1eff'], type(eris['cvcv']))):
      eriscvcv = lib.chkfile.load(eris['cvcv'].name, "eri_mo")#h5py.File(eris['cvcv'].name,'r')["eri_mo"]
    eris_sp['cvcv'] = eriscvcv.reshape(ncor, nvir, ncor, nvir)
    end = time.time()
    print('......production of INT took %10.2f sec' %(end-start))
    print('')

    # energy_core
    hcore  = mc.get_hcore()
    dmcore = np.dot(mo[:,:ncor], mo[:,:ncor].T)*2
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    energy_core = np.einsum('ij,ji', dmcore, hcore) \
                + np.einsum('ij,ji', dmcore, vj-0.5*vk) * .5

    # energyE0
    energyE0 = 1.0*np.einsum('ij,ij',     E1, eris_sp['h1eff'][ncor:nocc,ncor:nocc])\
             + 0.5*np.einsum('ijkl,ijkl', E2, eris['ppaa'][ncor:nocc,ncor:nocc,:,:].transpose(0,2,1,3))
    energyE0 += energy_core
    energyE0 += mc.mol.energy_nuc()

    print("Energy_nuc  = %13.8f"%(mc.mol.energy_nuc()))
    print("Energy_core = %13.8f"%(energy_core))
    print("Energy      = %13.8f"%(energyE0))
    print("")

    # offdiagonal warning
    offdiagonal = 0.0
    for k in range(ncor):
      for l in range(ncor):
        if(k != l):
          offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    for k in range(nocc, norb):
      for l in range(nocc,norb):
        if(k != l):
          offdiagonal = max(abs(offdiagonal), abs(eris_sp['h1eff'][k,l] ))
    if (abs(offdiagonal) > 1e-6):
      print("WARNING: Have to use natural orbitals from CAASCF")
      print("         offdiagonal elements: {:13.6f}".format(offdiagonal))
      print("")

    # Write out ingredients to intfolder
    # 2 "C"
    start = time.time()
    print("Basic ingredients written to "+intfolder,nfro,ncor,nocc,norb)
    np.save(intfolder+"W:ccae", np.asfortranarray(eris['pacv'][nfro:ncor,     :    , nfro:    ,     :    ].transpose(0,2,1,3)))
    np.save(intfolder+"W:eecc", np.asfortranarray(eris_sp['cvcv'][nfro: ,     :    , nfro:    ,     :    ].transpose(1,3,0,2)))

    # 2 "A"
    np.save(intfolder+"W:caac", np.asfortranarray(eris['papa'][nfro:ncor,     :    , nfro:ncor,     :    ].transpose(0,3,1,2)))
    np.save(intfolder+"W:ccaa", np.asfortranarray(eris['papa'][nfro:ncor,     :    , nfro:ncor,     :    ].transpose(0,2,1,3)))
    np.save(intfolder+"W:aeca", np.asfortranarray(eris['papa'][nfro:ncor,     :    , nocc:    ,     :    ].transpose(1,2,0,3)))
    np.save(intfolder+"W:eeaa", np.asfortranarray(eris['papa'][nocc:    ,     :    , nocc:    ,     :    ].transpose(0,2,1,3)))
    np.save(intfolder+"W:aaaa", np.asfortranarray(eris['ppaa'][ncor:nocc, ncor:nocc,     :    ,     :    ].transpose(0,2,1,3)))
    np.save(intfolder+"W:eaca", np.asfortranarray(eris['ppaa'][nocc:    , nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))
    np.save(intfolder+"W:caca", np.asfortranarray(eris['ppaa'][nfro:ncor, nfro:ncor,     :    ,     :    ].transpose(0,2,1,3)))

    # 2 "E"
    np.save(intfolder+"W:eeca", np.asfortranarray(eris['pacv'][nocc:    ,     :    , nfro:    ,     :    ].transpose(3,0,2,1)))

    end = time.time()
    print('......savings of INGREDIENTS took %10.2f sec' %(end-start))
    print("")

    return norb, energyE0


def write_ic_inputs(nelec, ncor, ncas, nfro, ms2, type):
    methods = ['_CCVV', '_CCAV', '_ACVV']
    domains = ['eecc','ccae','eeca','ccaa','eeaa','caae']

    for method in methods:
        # Prepare Input
        f = open("%s.inp"%(type+method), 'w')
        f.write('method %s\n'%(type+method))
        f.write('orb-type spatial/MO\n')
        f.write('nelec %d\n'%(nelec+(ncor-nfro)*2))
        f.write('nact %d\n'%(nelec))
        f.write('nactorb %d\n'%(ncas))
        f.write('ms2 %d\n'%(ms2))
        f.write('int1e/fock int/int1eff.npy\n')
        if (type=='MRLCC'):
          f.write('int1e/coreh int/int1.npy\n')
        #f.write('E3  int/E3.npy\n')
        #f.write('E2  int/E2.npy\n')
        #f.write('E1  int/E1.npy\n')
        f.write('thr-den 1.000000e-05\n')
        f.write('thr-var 1.000000e-05\n')
        f.write('thr-trunc 1.000000e-04\n')
        f.close();
    sys.stdout.flush()



def _ERIS(mc, mo, method='incore'):
    nmo = mo.shape[1]
    ncor = mc.ncore
    ncas = mc.ncas

    if ((method == 'outcore') or
        (mcscf.mc_ao2mo._mem_usage(ncor, ncas, nmo)[0] +
         nmo**4*2/1e6 > mc.max_memory*.9) or
        (mc._scf._eri is None)):
        ppaa, papa, pacv, cvcv = \
                trans_e1_outcore(mc, mo, max_memory=mc.max_memory,
                                 verbose=mc.verbose)
    else:
        ppaa, papa, pacv, cvcv = trans_e1_incore(mc, mo)

    dmcore = np.dot(mo[:,:ncor], mo[:,:ncor].T)
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(np.dot, (mo.T, vj*2-vk, mo))

    eris = {}
    eris['vhf_c'] = vhfcore
    eris['ppaa'] = ppaa
    eris['papa'] = papa
    eris['pacv'] = pacv
    eris['cvcv'] = cvcv
    eris['h1eff'] = reduce(np.dot, (mo.T, mc.get_hcore(), mo)) + vhfcore
    return eris


# see mcscf.mc_ao2mo
def trans_e1_incore(mc, mo):
    eri_ao = mc._scf._eri
    ncor = mc.ncore
    ncas = mc.ncas
    nmo = mo.shape[1]
    nocc = ncor + ncas
    nav = nmo - ncor
    eri1 = ao2mo.incore.half_e1(eri_ao, (mo[:,:nocc],mo[:,ncor:]),
                                      compact=False)
    load_buf = lambda r0,r1: eri1[r0*nav:r1*nav]
    ppaa, papa, pacv, cvcv = _trans(mo, ncor, ncas, load_buf)
    return ppaa, papa, pacv, cvcv


def trans_e1_outcore(mc, mo, max_memory=None, ioblk_size=256, tmpdir=None,
                     verbose=0):
    time0 = (time.process_time(), time.time())
    mol = mc.mol
    log = logger.Logger(mc.stdout, verbose)
    ncor = mc.ncore
    ncas = mc.ncas
    nao, nmo = mo.shape
    nao_pair = nao*(nao+1)//2
    nocc = ncor + ncas
    nvir = nmo - nocc
    nav = nmo - ncor

    if tmpdir is None:
        tmpdir = lib.param.TMPDIR
    swapfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    ao2mo.outcore.half_e1(mol, (mo[:,:nocc],mo[:,ncor:]), swapfile.name,
                                max_memory=max_memory, ioblk_size=ioblk_size,
                                verbose=log, compact=False)

    fswap = h5py.File(swapfile.name, 'r')
    klaoblks = len(fswap['0'])
    def load_buf(r0,r1):
        if mol.verbose >= logger.DEBUG1:
            time1[:] = logger.timer(mol, 'between load_buf',
                                              *tuple(time1))
        buf = np.empty(((r1-r0)*nav,nao_pair))
        col0 = 0
        for ic in range(klaoblks):
            dat = fswap['0/%d'%ic]
            col1 = col0 + dat.shape[1]
            buf[:,col0:col1] = dat[r0*nav:r1*nav]
            col0 = col1
        if mol.verbose >= logger.DEBUG1:
            time1[:] = logger.timer(mol, 'load_buf', *tuple(time1))
        return buf
    time0 = logger.timer(mol, 'halfe1', *time0)
    time1 = [time.process_time(), time.time()]
    ao_loc = np.array(mol.ao_loc_nr(), dtype=np.int32)
    cvcvfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    with h5py.File(cvcvfile.name,"w") as f5:	#Edit JoKurian - Earlier no "w"
        cvcv = f5.create_dataset('eri_mo', (ncor*nvir,ncor*nvir), 'f8')
        ppaa, papa, pacv = _trans(mo, ncor, ncas, load_buf, cvcv, ao_loc)[:3]
    time0 = logger.timer(mol, 'trans_cvcv', *time0)
    fswap.close()
    return ppaa, papa, pacv, cvcvfile


def _trans(mo, ncor, ncas, fload, cvcv=None, ao_loc=None):
    nao, nmo = mo.shape
    nocc = ncor + ncas
    nvir = nmo - nocc
    nav = nmo - ncor

    if cvcv is None:
        cvcv = np.zeros((ncor*nvir,ncor*nvir))
    pacv = np.empty((nmo,ncas,ncor*nvir))
    aapp = np.empty((ncas,ncas,nmo*nmo))
    papa = np.empty((nmo,ncas,nmo*ncas))
    vcv = np.empty((nav,ncor*nvir))
    apa = np.empty((ncas,nmo*ncas))
    vpa = np.empty((nav,nmo*ncas))
    app = np.empty((ncas,nmo*nmo))
    for i in range(ncor):
        buf = fload(i, i+1)
        klshape = (0, ncor, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        cvcv[i*nvir:(i+1)*nvir] = vcv[ncas:]
        pacv[i] = vcv[:ncas]

        klshape = (0, nmo, ncor, nocc)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=apa, ao_loc=ao_loc)
        papa[i] = apa
    for i in range(ncas):
        buf = fload(ncor+i, ncor+i+1)
        klshape = (0, ncor, nocc, nmo)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vcv, ao_loc=ao_loc)
        pacv[ncor:,i] = vcv

        klshape = (0, nmo, ncor, nocc)
        _ao2mo.nr_e2(buf, mo, klshape,
                      aosym='s4', mosym='s1', out=vpa, ao_loc=ao_loc)
        papa[ncor:,i] = vpa

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2(buf[:ncas], mo, klshape,
                      aosym='s4', mosym='s1', out=app, ao_loc=ao_loc)
        aapp[i] = app
    #lib.transpose(aapp.reshape(ncas**2, -1), inplace=True)
    ppaa = lib.transpose(aapp.reshape(ncas**2,-1))
    return (ppaa.reshape(nmo,nmo,ncas,ncas), papa.reshape(nmo,ncas,nmo,ncas),
            pacv.reshape(nmo,ncas,ncor,nvir), cvcv)

