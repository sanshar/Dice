import numpy
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc

r = 0.958
theta = 104.4776 * numpy.pi / 180.
atomstring = f'O 0. 0. 0.; H {r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.; H {-r * numpy.sin(theta/2)}  {r * numpy.cos(theta/2)} 0.'
mol = gto.M(
    atom=atomstring,
    basis='ccpvdz',
    verbose=4,
    symmetry=0,
    spin=0)
mf = scf.RHF(mol)
mf.kernel()

# freezing core and writing integrals
ncore = 1
norbAct = mol.nao - ncore
nelecAct = mol.nelectron - 2*ncore
mc = mcscf.CASSCF(mf, norbAct, nelecAct)
mc.mo_coeff = mf.mo_coeff
moCore = mc.mo_coeff[:,:ncore]
core_dm = 2 * moCore.dot(moCore.T)
corevhf = mc.get_veff(mol, core_dm)
energy_core = mol.energy_nuc()
energy_core += numpy.einsum('ij,ji', core_dm, mc.get_hcore())
energy_core += numpy.einsum('ij,ji', core_dm, corevhf) * .5
moAct = mc.mo_coeff[:, ncore:ncore + norbAct]
h1eff = moAct.T.dot(mc.get_hcore() + corevhf).dot(moAct)
eri = ao2mo.kernel(mol, moAct)
tools.fcidump.from_integrals('FCIDUMP', h1eff, eri, norbAct, nelecAct, energy_core)

# ccsd
mycc = cc.CCSD(mf)
mycc.frozen = ncore
mycc.kernel()
et = mycc.ccsd_t()
print('CCSD(T) energy', mycc.e_tot + et)

eip, cip = mycc.ipccsd(nroots=5)
print(f"eip:  {eip}")
