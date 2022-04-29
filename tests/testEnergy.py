#!/usr/bin/env python

import sys

def check_results(eRef, eTest, tol):

    if (abs(eRef - eTest) < tol):
        print("test passed")
    else:
        print("test failed")
        print("eRef = ", eRef)
        print("eTest = ", eTest)

def check_results_afqmc(eRef, eTest, wRef, wTest, tol):

    if ((abs(eRef - eTest)) < tol and (abs(wRef - wTest) < tol)):
        print("test passed")
    else:
        print("test failed")
        print("eRef = ", eRef)
        print("eTest = ", eTest)
        print("wRef = ", wRef)
        print("wTest = ", wTest)

def check_results_fciqmc_replica(eRef1, eRef2, eRefVar, eRefEN2,
                                 eTest1, eTest2, eTestVar, eTestEN2, tol):

    passed_1 = abs(eRef1 - eTest1) < tol
    passed_2 = abs(eRef2 - eTest2) < tol
    passed_var = abs(eRefVar - eTestVar) < tol
    passed_en2 = abs(eRefEN2 - eTestEN2) < tol

    if (passed_1 and passed_2 and passed_var and passed_en2):
        print("test passed")
    else:
        print("test failed")
        print("eRef1 = ", eRef1)
        print("eTest1 = ", eTest1)
        print("eRef2 = ", eRef2)
        print("eTest2 = ", eTest2)
        print("eRefVar = ", eRefVar)
        print("eTestVar = ", eTestVar)
        print("eRefEN2 = ", eRefEN2)
        print("eTestEN2 = ", eTestEN2)

def check_results_sp(determERef, determETest, stochERef, stochETest, tol):

    if (abs(determERef - determETest) < tol):
        print("determ test passed")
    else:
        print("determ test failed")
        print("determERef = ", determERef)
        print("determETest = ", determETest)

    if (abs(stochERef - stochETest) < tol):
        print("stoch test passed")
    else:
        print("stoch test failed")
        print("stochERef = ", stochERef)
        print("stochETest = ", stochETest)

if __name__ == '__main__':

    mc = sys.argv[1]
    tol = float(sys.argv[2])

    if mc == 'vmc' or mc == 'gfmc' or mc == 'ci':
        fh = open(mc+'.ref', 'r')
        for line in fh:
            pass
        eRef = float(line.split()[1])

        fh = open(mc+'.out', 'r')
        for line in fh:
            pass
        eTest = float(line.split()[1])

        check_results(eRef, eTest, tol)

    elif mc == 'afqmc':
        fh = open('samples.ref', 'r')
        for line in fh:
            pass
        wRef = float(line.split()[0])
        eRef = float(line.split()[1])

        fh = open('samples.dat', 'r')
        for line in fh:
            pass
        wTest = float(line.split()[0])
        eTest = float(line.split()[1])

        check_results_afqmc(eRef, eTest, wRef, wTest, tol)

    elif mc == 'nevpt' or mc == 'nevpt_print' or mc == 'nevpt_read':
        fh = open(mc+'.ref', 'r')
        for line in fh:
            pass
        eRef = float(line.split()[-1])

        fh = open(mc+'.out', 'r')
        for line in fh:
            pass
        eTest = float(line.split()[-1])

        check_results(eRef, eTest, tol)

    elif mc == 'single_perturber':
        fh = open('nevpt.ref', 'r')
        for line in fh:
            pass
        determERef = float(line.split()[-1])

        fh = open('nevpt.out', 'r')
        for line in fh:
            pass
        determETest = float(line.split()[-1])

        fh = open('stoch_samples.ref', 'r')
        for line in fh:
            pass
        stochERef = float(line.split()[1])

        fh = open('stoch_samples_0.dat', 'r')
        for line in fh:
            pass
        stochETest = float(line.split()[1])

        check_results_sp(determERef, determETest, stochERef, stochETest, tol)

    elif mc == 'fciqmc':
        last_line = ''
        fh = open(mc+'.benchmark', 'r')
        for line in fh:
            if '#' not in line:
                last_line = line

        eRef = float(last_line.split()[5]) / float(last_line.split()[6])

        fh = open(mc+'.out', 'r')
        for line in fh:
            if '#' not in line:
                last_line = line

        eTest = float(last_line.split()[5]) / float(last_line.split()[6])

        check_results(eRef, eTest, tol)

    elif mc == 'fciqmc_replica':
        last_line = ''
        fh = open('fciqmc.benchmark', 'r')

        # true if EN2 energies are being estimated in this job
        doingEN2 = False

        for line in fh:
            if '#' not in line:
                last_line = line
            else:
                if 'EN2' in line:
                    doingEN2 = True

        eRef1 = float(last_line.split()[5]) / float(last_line.split()[6])
        eRef2 = float(last_line.split()[9]) / float(last_line.split()[10])
        eRefVar = float(last_line.split()[11]) / float(last_line.split()[12])

        eRefEN2 = 0.0
        if doingEN2:
            eRefEN2 = float(last_line.split()[13]) / float(last_line.split()[12])

        fh = open('fciqmc.out', 'r')
        for line in fh:
            if '#' not in line:
                last_line = line

        eTest1 = float(last_line.split()[5]) / float(last_line.split()[6])
        eTest2 = float(last_line.split()[9]) / float(last_line.split()[10])
        eTestVar = float(last_line.split()[11]) / float(last_line.split()[12])

        eTestEN2 = 0.0
        if doingEN2:
            eTestEN2 = float(last_line.split()[13]) / float(last_line.split()[12])

        check_results_fciqmc_replica(eRef1, eRef2, eRefVar, eRefEN2,
                                     eTest1, eTest2, eTestVar, eTestEN2, tol)

    elif mc == 'fciqmc_trial':
        last_line = ''
        fh = open('fciqmc.benchmark', 'r')
        for line in fh:
            if '#' not in line:
                last_line = line

        eRef = float(last_line.split()[7]) / float(last_line.split()[8])

        fh = open('fciqmc.out', 'r')
        for line in fh:
            if '#' not in line:
                last_line = line

        eTest = float(last_line.split()[7]) / float(last_line.split()[8])

        check_results(eRef, eTest, tol)
