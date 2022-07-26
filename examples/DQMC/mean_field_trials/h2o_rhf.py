# H2O ground state

from pyscf import gto, scf
import QMCUtils
import numpy

# mean field calculation
r = 0.9578
theta = 104.5078 * numpy.pi / 180.
atomstring = f'O 0. 0. 0.; H {r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.; H {-r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.'
mol = gto.M(atom = atomstring, basis = '6-31g', verbose = 3, symmetry=1)
mf = scf.RHF(mol)
mf.kernel()

# afqmc calculation
QMCUtils.run_afqmc(mf, norb_frozen = 1)

