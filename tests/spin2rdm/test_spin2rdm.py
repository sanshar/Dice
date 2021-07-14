import sys, os
import numpy as np
from pyscf import gto, scf, mcscf, lib
from pyscf.shciscf import shci
from pyscf.shciscf.shci import make_sched
from pyscf.shciscf import settings as shci_settings


def read_Dice2RDM(dice2RDMName):
    with open(dice2RDMName) as f:
        content = f.readlines()

    norbs = int(content[0].split()[0])
    dice2RDM = np.zeros((norbs,) * 4)

    for i in range(1, len(content)):
        c0, c1, d1, d0, val = content[i].split()
        dice2RDM[int(c0), int(c1), int(d1), int(d0)] = float(val)

    return dice2RDM


def read_dice_spin_2rdm(mc, root, nelec):
    # Read in spin2RDM
    file2pdm = "spin2RDM.%d.%d.txt" % (root, root)
    filename = os.path.join(mc.fcisolver.scratchDirectory, file2pdm)

    spin_2rdm = read_Dice2RDM(filename)
    dm2aa = spin_2rdm[::2, ::2, ::2, ::2].transpose(0, 2, 1, 3)
    dm2ab = spin_2rdm[::2, 1::2, ::2, 1::2].transpose(0, 2, 1, 3)
    dm2bb = spin_2rdm[1::2, 1::2, 1::2, 1::2].transpose(0, 2, 1, 3)

    # We're summing over alpha and beta so we need to divide by their sum
    # fmt: off
    dm1a = np.einsum("ikjj", spin_2rdm.transpose(0, 2, 1, 3))[::2, ::2] / (nelec[0]+nelec[1] - 1.0)
    dm1b = np.einsum("jjik", spin_2rdm.transpose(0, 2, 1, 3))[1::2, 1::2] / (nelec[0]+nelec[1] - 1.0)
    # fmt: on
    # print("dm1a nelec", np.einsum("ii", dm1a))
    # print("dm1b nelec", np.einsum("ii", dm1b))
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)


def restricted_spin_square(mc, root: int, nelec=None) -> float:
    """Calculate <S^2> for a CAS-like object with restricted orbitals.

    Parameters
    ----------
    mc :
        A CAS-like object.
    root : int
        The root we want <S^2>, use 0-based indexing.
    nelec : [type], optional
        Specify the desired nelec tuple, e.g. (8,6). By default None

    Returns
    -------
    float
        <S^2> for the root of the system in question.

    """

    lib.logger.note(mc.mol, f"\tCalculating <S^2> for CI-type |\u03A8> root: {root}")

    # Input checking
    if nelec is None:
        if isinstance(mc.nelecas, int):
            nelec = (mc.nelecas // 2, mc.nelecas // 2)
        else:
            nelec = mc.nelecas
        # print(f"Picking nelec = {nelec}")
    else:
        if not isinstance(nelec, tuple):
            raise ValueError(
                "nelec must be a tuple with size of 2. E.g. nelec = (6,4)."
            )

    if isinstance(mc.fcisolver, shci.SHCI):
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = read_dice_spin_2rdm(mc, root, nelec)
    else:
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = mc.fcisolver.make_rdm12s(
            mc.ci[root], mc.ncas, mc.nelecas
        )

    # Actual Calculation
    ovlpaa = ovlpab = ovlpba = ovlpbb = np.diag(np.ones(mc.ncas))

    # Make sure <S_z^2> from the RDMs matches ((alpha-beta)/2.0)**2
    # if ovlp=1, ssz = (neleca-nelecb)**2 * .25
    ssz = (
        np.einsum("ijkl,ij,kl->", dm2aa, ovlpaa, ovlpaa)
        - np.einsum("ijkl,ij,kl->", dm2ab, ovlpaa, ovlpbb)
        + np.einsum("ijkl,ij,kl->", dm2bb, ovlpbb, ovlpbb)
        - np.einsum("ijkl,ij,kl->", dm2ab, ovlpaa, ovlpbb)
    ) * 0.25
    ssz += (
        np.einsum("ji,ij->", dm1a, ovlpaa) + np.einsum("ji,ij->", dm1b, ovlpbb)
    ) * 0.25
    # End testing

    dm2abba = -dm2ab.transpose(0, 3, 2, 1)  # alpha^+ beta^+ alpha beta
    dm2baab = -dm2ab.transpose(2, 1, 0, 3)  # beta^+ alpha^+ beta alpha

    # Calculate contribution from S_- and S_+
    ssxy = (
        np.einsum("ijkl,ij,kl->", dm2baab, ovlpba, ovlpab)
        + np.einsum("ijkl,ij,kl->", dm2abba, ovlpab, ovlpba)
        + np.einsum("ji,ij->", dm1a, ovlpaa)
        + np.einsum("ji,ij->", dm1b, ovlpbb)
    ) * 0.5

    sz = (nelec[0] - nelec[1]) / 2.0
    szsz = sz * sz
    ss = szsz + ssxy

    # Warn of if things aren't as we'd expect
    if abs(szsz - ssz) > 1e-10:
        lib.logger.warn(mc.mol, f"\t<S_z> numerical error {abs(szsz - ssz):.3e}")

    lib.logger.note(
        mc.mol, f"\t<S^2> = {ss:.3f}    <S_z^2> = {szsz:.3f}    <S_+S_-> = {ssxy:.3f}"
    )

    return ss


def print_test(mc: mcscf.CASSCF, test_name: str, error: float, tol: float = 1e-12):
    """Testing utility"""
    if error > tol:
        lib.logger.note(mc.mol, f"\t{test_name} error {error:.3e} > {tol:.1e}")
        lib.logger.note(mc.mol, f"\t\033[91mFailed\033[00m {test_name} Test....")
        return 1
    else:
        lib.logger.note(mc.mol, f"\t\033[92mPassed\033[00m {test_name} Test....")
        return 0


def writeSHCIConfFile(SHCI, nelec, Restart):
    """This function overwrites the SHCI member function to write a Dice
    input file and makes sure we add the `DoSpinRDM` flag.
    """
    # print("USING CUSTOM writeSHCIConfFile function")
    confFile = os.path.join(SHCI.runtimeDir, SHCI.configFile)

    f = open(confFile, "w")

    # Reference determinant section
    f.write("#system\n")
    f.write("nocc %i\n" % (nelec[0] + nelec[1]))
    if SHCI.__class__.__name__ == "FakeCISolver":
        for i in range(nelec[0]):
            f.write("%i " % (2 * i))
        for i in range(nelec[1]):
            f.write("%i " % (2 * i + 1))
    else:
        if SHCI.initialStates is not None:
            for i in range(len(SHCI.initialStates)):
                for j in SHCI.initialStates[i]:
                    f.write("%i " % (j))
                if i != len(SHCI.initialStates) - 1:
                    f.write("\n")
        elif SHCI.irrep_nelec is None:
            for i in range(int(nelec[0])):
                f.write("%i " % (2 * i))
            for i in range(int(nelec[1])):
                f.write("%i " % (2 * i + 1))
        else:
            from pyscf import symm
            from pyscf.dmrgscf import dmrg_sym
            from pyscf.symm.basis import DOOH_IRREP_ID_TABLE

            if SHCI.groupname is not None and SHCI.orbsym is not []:
                orbsym = dmrg_sym.convert_orbsym(SHCI.groupname, SHCI.orbsym)
            else:
                orbsym = [1] * norb
            done = []
            for k, v in SHCI.irrep_nelec.items():

                irrep, nalpha, nbeta = (
                    [dmrg_sym.irrep_name2id(SHCI.groupname, k)],
                    v[0],
                    v[1],
                )

                for i in range(len(orbsym)):  # loop over alpha electrons
                    if orbsym[i] == irrep[0] and nalpha != 0 and i * 2 not in done:
                        done.append(i * 2)
                        f.write("%i " % (i * 2))
                        nalpha -= 1
                    if orbsym[i] == irrep[0] and nbeta != 0 and i * 2 + 1 not in done:
                        done.append(i * 2 + 1)
                        f.write("%i " % (i * 2 + 1))
                        nbeta -= 1
                if nalpha != 0:
                    print(
                        "number of irreps %s in active space = %d" % (k, v[0] - nalpha)
                    )
                    print("number of irreps %s alpha electrons = %d" % (k, v[0]))
                    exit(1)
                if nbeta != 0:
                    print(
                        "number of irreps %s in active space = %d" % (k, v[1] - nbeta)
                    )
                    print("number of irreps %s beta  electrons = %d" % (k, v[1]))
                    exit(1)
    f.write("\nend\n")

    # Handle different cases for FCIDUMP file names/paths
    f.write("orbitals {}\n".format(os.path.join(SHCI.runtimeDir, SHCI.integralFile)))

    f.write("nroots %r\n" % SHCI.nroots)
    if SHCI.mol.symmetry and SHCI.mol.groupname:
        f.write(f"pointGroup {SHCI.mol.groupname.lower()}\n")
    if hasattr(SHCI, "wfnsym"):
        f.write(f"irrep {SHCI.wfnsym}\n")

    # Variational Keyword Section
    f.write("\n#variational\n")
    if not Restart:
        schedStr = make_sched(SHCI)
        f.write(schedStr)
    else:
        f.write("schedule\n")
        f.write("%d  %g\n" % (0, SHCI.sweep_epsilon[-1]))
        f.write("end\n")

    f.write("davidsonTol %g\n" % SHCI.davidsonTol)
    f.write("dE %g\n" % SHCI.dE)

    # Sets maxiter to 6 more than the last iter in sweep_iter[] if restarted.
    if not Restart:
        f.write("maxiter %i\n" % (SHCI.sweep_iter[-1] + 6))
    else:
        f.write("maxiter 10\n")
        f.write("fullrestart\n")

    # Perturbative Keyword Section
    f.write("\n#pt\n")
    if SHCI.stochastic == False:
        f.write("deterministic \n")
    else:
        f.write("nPTiter %d\n" % SHCI.nPTiter)
    f.write("epsilon2 %g\n" % SHCI.epsilon2)
    f.write("epsilon2Large %g\n" % SHCI.epsilon2Large)
    f.write("targetError %g\n" % SHCI.targetError)
    f.write("sampleN %i\n" % SHCI.sampleN)

    # Miscellaneous Keywords
    f.write("\n#misc\n")
    f.write("io \n")
    if SHCI.scratchDirectory != "":
        if not os.path.exists(SHCI.scratchDirectory):
            os.makedirs(SHCI.scratchDirectory)
        f.write("prefix %s\n" % (SHCI.scratchDirectory))
    if SHCI.DoRDM:
        f.write("DoOneRDM\n")
        f.write("DoSpinOneRDM\n")
        f.write("DoSpinRDM\n")
        f.write("DoRDM\n")
    for line in SHCI.extraline:
        f.write("%s\n" % line)

    f.write("\n")  # SHCI requires that there is an extra line.
    f.close()


if __name__ == "__main__":

    shci.writeSHCIConfFile = writeSHCIConfFile
    shci_settings.SHCIEXE = "../../Dice"
    verbose = 3

    ch2 = {
        "cas": (6, 6),
        "spin": 0,
        "nroots": 6,
        "atom": "H 0 0 -1; C 0 0 0; H 0 0 1;",
        "name": "CH2",
    }
    cn = {
        "cas": (8, 9),
        "spin": 1,
        "nroots": 5,
        "atom": "C 0 0 0; N 0 0 1",
        "name": "CN",
    }
    c2 = {
        "cas": (8, 8),
        "spin": 0,
        "nroots": 10,
        "atom": "C 0 0 0;C 0 0 1",
        "name": "C2",
    }
    o2 = {
        "cas": (8, 12),
        "spin": 0,
        "nroots": 4,
        "atom": "O 0 0 0; O 0 0 1",
        "name": "O2",
    }

    systems = [
        ch2,
        o2,
        cn,
        c2,
    ]

    for system in systems:
        ncas, nelecas = system["cas"]
        spin = system["spin"]
        atom = system["atom"]
        name = system["name"]
        nroots = system["nroots"]

        mol = gto.M(
            atom=atom,
            basis="ccpvdz",
            verbose=verbose,
            spin=spin,
            symmetry=True,
            output=f"{name}.out",
        )
        mf = scf.RHF(mol).run()
        # noons, natorbs = mcscf.addons.make_natural_orbitals(mf)
        # lib.logger.note(mol, f"UHF UNO Occ.: {noons}")
        mo = mf.mo_coeff.copy()

        trusted_mc = mcscf.CASCI(mf, ncas, nelecas)
        trusted_mc.fcisolver.nroots = nroots
        # trusted_mc.kernel(natorbs)
        trusted_mc.kernel(mo)

        mf.mo_coeff = mo

        mc = mcscf.CASCI(mf, ncas, nelecas)
        mc.fcisolver = shci.SHCI(mol)
        mc.fcisolver.sweep_iter = [0, 3, 6]
        mc.fcisolver.sweep_epsilon = [1e-10] * 3
        mc.fcisolver.nroots = nroots
        # mc.kernel(natorbs)
        mc.kernel(mo)

        if isinstance(mc.nelecas, int):
            nelec = (mc.nelecas // 2, mc.nelecas // 2)
        else:
            nelec = mc.nelecas

        # Keep track of errors
        n_fail = 0

        # Test <S^2> for each root
        for ni in range(nroots):
            lib.logger.note(mol, "##############")
            lib.logger.note(mol, f"#   ROOT {ni}   #")
            lib.logger.note(mol, "##############")
            (dice_dm1a, dice_dm1b), (
                dice_dm2aa,
                dice_dm2ab,
                dice_dm2bb,
            ) = read_dice_spin_2rdm(mc, ni, nelec)
            (pyscf_dm1a, pyscf_dm1b), (
                pyscf_dm2aa,
                pyscf_dm2ab,
                pyscf_dm2bb,
            ) = trusted_mc.fcisolver.make_rdm12s(
                trusted_mc.ci[ni], trusted_mc.ncas, trusted_mc.nelecas
            )

            # Testing
            dm1a_err = np.linalg.norm(dice_dm1a - pyscf_dm1a) / dice_dm1a.size
            dm1b_err = np.linalg.norm(dice_dm1b - pyscf_dm1b) / dice_dm1b.size
            dm2aa_err = np.linalg.norm(dice_dm2aa - pyscf_dm2aa) / dice_dm2aa.size
            dm2ab_err = np.linalg.norm(dice_dm2ab - pyscf_dm2ab) / dice_dm2ab.size
            dm2bb_err = np.linalg.norm(dice_dm2bb - pyscf_dm2bb) / dice_dm2bb.size

            n_fail += print_test(mc, f"{name} root={ni} DM1a", dm1a_err, tol=1e-5)
            n_fail += print_test(mc, f"{name} root={ni} DM1b", dm1b_err, tol=1e-5)
            n_fail += print_test(mc, f"{name} root={ni} DM2aa", dm2aa_err, tol=1e-5)
            n_fail += print_test(mc, f"{name} root={ni} DM2ab", dm2ab_err, tol=1e-5)
            n_fail += print_test(mc, f"{name} root={ni} DM2bb", dm2bb_err, tol=1e-5)

            pyscf_ss = restricted_spin_square(trusted_mc, ni)
            dice_ss = restricted_spin_square(mc, ni)
            ss_err = abs(pyscf_ss - dice_ss)
            n_fail += print_test(mc, f"{name} root={ni} <S^2>", ss_err, tol=1e-7)

            # print()

        print(f"\nNumber of failed tests from {name:3s} = {n_fail} out of {6*nroots}\n")
        # print("\n")

    mc.fcisolver.cleanup_dice_files()
