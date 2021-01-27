import os
import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.shciscf import shci
import pandas as pd

spin = 0

mol = gto.Mole()
mol.atom = "C 0 0 -1; C 0 0 1; C 0 -1.5 -1; C 0 -1.5 1;"
mol.spin = spin
mol.charge = 0
mol.basis = "sto3g"
mol.verbose = 4
mol.symmetry = True
mol.output = "_make_integrals.out"
mol.build()
mf = scf.ROHF(mol).run()

ncas, nelecas = (12, 8)
cas = mcscf.CASSCF(mf, ncas, nelecas)
cas.kernel()

mc = shci.SHCISCF(mf, ncas, nelecas)
mc.fcisolver.sweep_iter = [0, 3]
mc.fcisolver.sweep_epsilon = [0, 0]
mc.fcisolver.scratchDirectory = "."
mc.fcisolver.wfnsym = cas.fcisolver.wfnsym
mc.kernel(cas.mo_coeff)


data = {"irreps": ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"], "energy": []}
for irrep in data["irreps"]:
    mycasci = mcscf.CASCI(mf, ncas, nelecas)
    mycasci.fcisolver.wfnsym = irrep
    # mycasci.fix_spin_(ss=0)
    mycasci.kernel(cas.mo_coeff)
    data["energy"].append(mycasci.e_tot)

#
df = pd.DataFrame(data)
df.to_csv(f"pyscf_energies_spin={spin}.csv", index=False)
print(data)

os.system(f"mv FCIDUMP ../integrals/c4_FCIDUMP_d2h_spin={spin}")
os.system("rm -f shci.e *.bkp *.txt *.dat")
