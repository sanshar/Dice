# c atom ground state

from pyscf import gto, scf
import QMCUtils

# mean field calculation
mol = gto.M(atom = "C 0 0 0", basis = '6-31g', verbose = 3, symmetry='dooh', spin=2)
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1g':(2,2), 'A1u':(0,0), 'E1ux':(1,0), 'E1uy':(1,0)}
mf.kernel()

# afqmc calculation
QMCUtils.run_afqmc(mf, norb_frozen = 1)

