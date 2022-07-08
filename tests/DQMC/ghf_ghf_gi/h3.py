from pyscf import gto, scf, tools, fci
import numpy as np
import scipy.linalg as la
import QMCUtils

a = 1.4
atomstring = f'''
H 0. 0. {a/(3**0.5)}
H {a/2} 0. {-a/2/(3**0.5)}
H {-a/2} 0. {-a/2/(3**0.5)}
'''

mol = gto.M(atom = atomstring, basis = 'sto-6g', verbose = 3, unit = 'angstrom', symmetry = 0, spin = 1)
norb = mol.nao

mf = scf.RHF(mol)
mf.kernel()
lmo = QMCUtils.localizeAllElectron(mf, method="lowdin")

cisolver = fci.FCI(mf)
e_fci, ci = cisolver.kernel()
print('e(FCI) = %.12f' % e_fci)

gmf = scf.GHF(mol)
staggered_mos = np.zeros((6,3))
staggered_mos[:3,0] = lmo[:,0]
staggered_mos[:3,1] = 0.5 * lmo[:,1]
staggered_mos[3:,1] = 3**0.5 / 2. * lmo[:,1]
staggered_mos[:3,2] = 0.5 * lmo[:,2]
staggered_mos[3:,2] = -3**0.5 / 2. * lmo[:,2]
dm = staggered_mos.dot(staggered_mos.T)
gmf.kernel(dm)

QMCUtils.prepAFQMC_gihf(mol, gmf)
#chol_vecs = QMCUtils.chunked_cholesky(mol, max_error=1e-5)
#nchol = chol_vecs.shape[0]
#chol = np.zeros((nchol, 2*norb, 2*norb))
#for i in range(nchol):
#  chol_i = chol_vecs[i].reshape(norb, norb)
#  chol_i = la.block_diag(chol_i, chol_i)
#  chol[i] = gmf.mo_coeff.T.dot(chol_i).dot(gmf.mo_coeff)
#hcore = mf.get_hcore()
#hcore = la.block_diag(hcore, hcore)
#h1e = gmf.mo_coeff.T.dot(hcore).dot(gmf.mo_coeff)
#enuc = mol.energy_nuc()
#nbasis = h1e.shape[-1]
#print(f'nelec: {mol.nelec}')
#print(f'nbasis: {nbasis}')
#print(f'chol.shape: {chol.shape}')
#chol = chol.reshape((-1, nbasis, nbasis))
#v0 = 0.5 * np.einsum('nik,njk->ij', chol, chol, optimize='optimal')
#h1e_mod = h1e - v0
#chol = chol.reshape((chol.shape[0], -1))
#QMCUtils.write_dqmc(h1e, h1e_mod, chol, sum(mol.nelec), nbasis, enuc, ms=mol.spin, filename='FCIDUMP_chol')

ghf_coeffs = np.eye(2*norb)
QMCUtils.writeMat(ghf_coeffs, "ghf.txt")
