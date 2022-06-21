# H2O ground state

from pyscf import gto, scf
import QMCUtils
import numpy

# this needs to be provided
vmc_root = "/projects/anma2640/VMC/master/VMC"

# mean field calculation
r = 0.9578
theta = 104.5078 * numpy.pi / 180.
atomstring = f'O 0. 0. 0.; H {r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.; H {-r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.'
mol = gto.M(atom = atomstring, basis = '6-31g', verbose = 3, symmetry=1)
mf = scf.RHF(mol)
mf.kernel()

# afqmc calculation
QMCUtils.run_afqmc(mf, vmc_root = vmc_root, norb_frozen = 1)

