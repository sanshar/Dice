#python2
import sys

mc = sys.argv[1]
tol = float(sys.argv[2])

fh = open(mc+'.ref', 'r')
for line in fh:
    pass

eRef = float(line.split()[1])

fh = open(mc+'.out', 'r')
for line in fh:
    pass

eTest = float(line.split()[1])

if(abs(eRef-eTest)<tol):
    print "test passed"
else:
    print "test failed"
    print "eRef=", eRef
    print "eTest=", eTest
