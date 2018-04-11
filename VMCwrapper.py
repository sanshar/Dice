import numpy as np
import climin
import sys
import climin.rmsprop
import climin.amsgrad
from subprocess import check_output, check_call, CalledProcessError

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
Restart = False
for line in f:
    linesp = line.split();
    if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0] == "correlator"):
        correlatorFile = linesp[2]
        correlatorSize = int(linesp[1])
	#print linesp, correlatorFile
        f2 = open(correlatorFile, 'r')
        for line2 in f2:
            if (line2.strip(' \n') != ''):
                numCorrelators += 1
    elif ( len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "restart"):
	Restart = True
    
numVars = numCorrelators*2**(2*correlatorSize)
emin = 1.e10

def d_loss_wrt_pars(wrt):
    global emin
    wrt.astype('float64').tofile("params.bin")
    try:
        cmd = ' '.join((mpiprefix, executable, inFile))
        cmd = "%s " % (cmd)
        #check_call(cmd, shell=True)
        out=check_output(cmd, shell=True).strip()
        print out
        sys.stdout.flush()
        e=float(out.split()[0])
    except CalledProcessError as err:
        raise err

    if e<emin:
        emin = 1.*e
        wrt.astype('float64').tofile("params_min.bin")
        eminA = np.asarray([emin])
        eminA.astype('float64').tofile("emin.bin")

    p = np.fromfile("grad.bin", dtype="float64")
    p.reshape(wrt.shape)
    return p


wrt = np.ones(shape=(numVars,))
if (Restart):
    wrt  = np.fromfile("params.bin", dtype = "float64")
    emin = np.fromfile("emin.bin", dtype = "float64")[0]

#wrt = np.fromfile("params.bin", dtype="float64")
#opt = climin.GradientDescent(wrt, d_loss_wrt_pars, step_rate=0.01, momentum=.95)
#opt = climin.rmsprop.RmsProp(wrt, d_loss_wrt_pars, step_rate=0.0001, decay=0.9)
opt = climin.amsgrad.Amsgrad(wrt, d_loss_wrt_pars, step_rate=0.001, decay_mom1=0.1, decay_mom2=0.001, momentum=0.0)

if (Restart):	
   opt.est_mom1_b = np.fromfile("moment1.bin", dtype="float64")
   opt.est_mom2_b = np.fromfile("moment2.bin", dtype="float64")

for info in opt:
    if info['n_iter'] >= 5000:
        break 
    opt.est_mom1_b.astype("float64").tofile("moment1.bin")
    opt.est_mom2_b.astype("float64").tofile("moment2.bin")

