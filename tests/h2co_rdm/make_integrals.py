from functools import reduce
import numpy as np
from pyscf import gto, scf, mcscf, fci
from pyscf.shciscf import shci

#
# Generate and Opt. Orbs with CASSCF
#

mol = gto.M(
    atom="""
 C                  0.00000000    0.00000000   -0.56221066
 H                  0.00000000   -0.95444767   -1.10110537
 H                  0.00000000    0.95444767   -1.10110537
 O                  0.00000000    0.00000000    0.69618930""",
    basis="ccpvdz",
    spin=0,
    symmetry=True,
)
mf = scf.RHF(mol).run()
ncas, nelecas = (8, 10)

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.fcisolver.conv_tol = 1e-14
mc.kernel(mc.mo_coeff)
dm1, dm2, dm3 = fci.rdm.make_dm123("FCI3pdm_kern_sf", mc.ci, mc.ci, ncas, nelecas)
dm1, dm2, dm3 = fci.rdm.reorder_dm123(dm1, dm2, dm3)
np.save("pyscf_1RDM.npy", dm1)
np.save("pyscf_2RDM.npy", dm2)
np.save("pyscf_3RDM.npy", dm3)

#
# Use orbitals for an approx. CASCI and create RDMs
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
print(mc.ci.shape)
print(dm2.shape)
print(np.einsum("ii", dm1))
print(np.einsum("ii", np.einsum("ikjj", dm2) / (nelecas - 1)))
