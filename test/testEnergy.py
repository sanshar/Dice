#python2
import sys

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
    
if(abs(eRef-eTest)<tol):
    print "test passed"
else:
    print "test failed"
    print "eRef=", eRef
    print "eTest=", eTest
