import numpy as np
import climin
import sys
import climin.rmsprop
import climin.amsgrad
from subprocess import check_call, CalledProcessError

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


mpiprefix = "mpirun "
executable = "~/apps/VMC/PythonInterface"
myargs = getopts(sys.argv)
if '-i' in myargs:
   inFile = myargs['-i']

if '-n' in myargs:
   numprocs = int(myargs['-n'])
   mpiprefix = "mpirun -n %i"%(numprocs)

f = open(inFile, 'r')
correlatorSize, numCorrelators = 0, 0
for line in f:
    linesp = line.split();
    if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0] == "correlator"):
        correlatorFile = linesp[2]
        correlatorSize = int(linesp[1])
	print linesp, correlatorFile
        f2 = open(correlatorFile, 'r')
        for line2 in f2:
            if (line2.strip(' \n') != ''):
                numCorrelators += 1
        break;

numVars = numCorrelators*2**(2*correlatorSize)
print numVars, numCorrelators, correlatorSize
def d_loss_wrt_pars(wrt):
    wrt.astype('float64').tofile("params.bin")
    try:
        cmd = ' '.join((mpiprefix, executable, inFile))
        cmd = "%s " % (cmd)
        check_call(cmd, shell=True)
    except CalledProcessError as err:
        raise err
    p = np.fromfile("grad.bin", dtype="float64")
    p.reshape(wrt.shape)
    return p

wrt = np.ones(shape=(numVars,))
#wrt = np.fromfile("params.bin", dtype="float64")
#opt = climin.GradientDescent(wrt, d_loss_wrt_pars, step_rate=0.01, momentum=.95)
#opt = climin.rmsprop.RmsProp(wrt, d_loss_wrt_pars, step_rate=0.0001, decay=0.9)
opt = climin.amsgrad.Amsgrad(wrt, d_loss_wrt_pars, step_rate=0.001, decay_mom1=0.1, decay_mom2=0.001, momentum=0.0)

for info in opt:
    if info['n_iter'] >= 5000:
        break
    

