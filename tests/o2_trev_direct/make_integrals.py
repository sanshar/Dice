from functools import reduce
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.shciscf import shci

#
# Generate and Opt. Orbs with CASSCF
#
mol = gto.M(atom="O 0 0 0; O 0 0 1.", basis="ccpvdz", spin=2, symmetry=True)

mf = scf.RHF(mol).newton()
mf.kernel()

ncas, nelecas = (8, 12)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver.conv_tol = 1e-14
mc.kernel(mc.mo_coeff)
dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
np.save("pyscf_2RDM.npy", dm2)

#
# Use orbitals for an approx CASCI and create FCIDUMP for Dice
#
mc2 = mcscf.CASCI(mf, ncas, nelecas)
mc2.fcisolver = shci.SHCI(mol)
mc2.fcisolver.sweep_iter = [0]
mc2.fcisolver.sweep_epsilon = [1e-10]
mc2.fcisolver.scratchDirectory = "."
mc2.kernel(mc.mo_coeff)


#
# Helpful Outputs
#
print(dm2.shape)
print(np.einsum("ii", dm1))
print(np.einsum("ii", np.einsum("ikjj", dm2) / (nelecas - 1)))
