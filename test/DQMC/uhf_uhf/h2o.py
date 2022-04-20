import numpy
from pyscf import gto, scf
import QMCUtils

# build your molecule
r = 0.9578
theta = 104.5078 * numpy.pi / 180.
atomstring = f'O 0. 0. 0.; H {r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.; H {-r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.'
mol = gto.M(atom = atomstring, basis='6-31g', unit='angstrom', charge=1, spin=1, symmetry=1)
mf = scf.RHF(mol)
mf.kernel()

# uhf
umf = scf.UHF(mol)
umf.kernel()

# afqmc
print("Preparing AFQMC calculation")
overlap = mf.get_ovlp(mol)
norb = mol.nao
uhfCoeffs = numpy.empty((norb, 2*norb))
uhfCoeffs[:,:norb] = mf.mo_coeff.T.dot(overlap).dot(umf.mo_coeff[0])
uhfCoeffs[:,norb:] = mf.mo_coeff.T.dot(overlap).dot(umf.mo_coeff[1])
QMCUtils.writeMat(uhfCoeffs, "uhf.txt")

QMCUtils.prepAFQMC(mol, mf)
QMCUtils.write_afqmc_input(seed = 4321, left="uhf", right="uhf", nwalk=20, stochasticIter=50, choleskyThreshold=1.e-3, fname="afqmc.json")
