#!/usr/bin/env python

import sys

def check_results(eRef, eTest, tol):

    if (abs(eRef - eTest) < tol):
        print "test passed"
    else:
        print "test failed"
        print "eRef = ", eRef
        print "eTest = ", eTest

def check_results_sp(determERef, determETest, stochERef, stochETest, tol):

    if (abs(determERef - determETest) < tol):
        print "determ test passed"
    else:
        print "determ test failed"
        print "determERef = ", determERef
        print "determETest = ", determETest

    if (abs(stochERef - stochETest) < tol):
        print "stoch test passed"
    else:
        print "stoch test failed"
        print "stochERef = ", stochERef
        print "stochETest = ", stochETest

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
    
    elif mc == 'nevpt':
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
    
    elif 'fciqmc' in mc:
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
