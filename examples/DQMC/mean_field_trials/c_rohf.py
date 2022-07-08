# c atom ground state

from pyscf import gto, scf
import QMCUtils

# this needs to be provided
vmc_root = "/projects/anma2640/VMC/master/VMC"

# mean field calculation
mol = gto.M(atom = "C 0 0 0", basis = '6-31g', verbose = 3, symmetry='dooh', spin=2)
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1g':(2,2), 'A1u':(0,0), 'E1ux':(1,0), 'E1uy':(1,0)}
mf.kernel()

# afqmc calculation
QMCUtils.run_afqmc(mf, vmc_root = vmc_root, norb_frozen = 1)

