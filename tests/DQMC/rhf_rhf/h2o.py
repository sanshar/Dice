import numpy
from pyscf import gto, scf
import QMCUtils

# build your molecule
r = 0.9578
theta = 104.5078 * numpy.pi / 180.
atomstring = f'O 0. 0. 0.; H {r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.; H {-r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.'
mol = gto.M(atom = atomstring, basis='6-31g', unit='angstrom', symmetry=1)
mf = scf.RHF(mol)
mf.kernel()

# afqmc
print("Preparing AFQMC calculation")
rhfCoeffs = numpy.eye(mol.nao)
QMCUtils.writeMat(rhfCoeffs, "rhf.txt")

QMCUtils.prepAFQMC(mol, mf)
QMCUtils.write_afqmc_input(seed = 4321, left="rhf", right="rhf", nwalk=20, stochasticIter=50, choleskyThreshold=1.e-3, fname="afqmc.json")
