import numpy as np
import sys, os, climin.amsgrad, scipy
from functools import reduce
from subprocess import check_output, check_call, CalledProcessError

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts


m_stepsize = 0.001
m_decay_mom1 = 0.1
m_decay_mom2 = 0.001

mpiprefix = " mpirun "
executable = "/Users/sandeepsharma/Academics/Programs/VMC/bin/PythonInterface"
myargs = getopts(sys.argv)
if '-i' in myargs:
   inFile = myargs['-i']

if '-n' in myargs:
   numprocs = int(myargs['-n'])
   mpiprefix = "mpirun -n %i"%(numprocs)

if '-stepsize' in myargs:
   m_stepsize = float(myargs['-stepsize'])

if '-decay1' in myargs:
   m_decay_mom1 = float(myargs['-decay1'])

if '-decay2' in myargs:
   m_decay_mom2 = float(myargs['-decay2'])


f = open(inFile, 'r')
correlatorSize, numCorrelators = 0, 0
Restart = False
ciExpansion = []
doHessian = False
maxIter = 1000
numVars = 0
UHF = False

print "#*********INPUT FILE"
for line in f:
    linesp = line.split();
    print "#", line,
    #read the correlator file to determine the number of jastrow factors
    if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "correlator"):
        correlatorFile = linesp[2]
        correlatorSize = int(linesp[1])
        numCorrelators = 0
	#print linesp, correlatorFile
        f2 = open(correlatorFile, 'r')
        for line2 in f2:
            if (line2.strip(' \n') != ''):
                numCorrelators += 1
        numVars += numCorrelators*2**(2*correlatorSize)
    elif ( len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "restart"):
	    Restart = True
    #read the determinant file to see the number of determinants
    if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "determinants"):
        determinantFile = linesp[1]
        f2 = open(determinantFile, 'r')
        for line2 in f2:
            if (line2.strip(' \n') != ''):
                tok = line2.split()
                ciExpansion.append(float(tok[0]))
    if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "uhf"):
        UHF = True
        
    if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "dohessian"):
        doHessian = True
    if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "maxiter"):
        maxIter = int(linesp[1])
        
print "#*********END OF INPUT FILE"
print "#opt-params"
print "#stepsize   : %f"%(m_stepsize)
print "#decay_mom1 : %f"%(m_decay_mom1)
print "#decay_mom2 : %f"%(m_decay_mom2)
print "#**********"

if (len(ciExpansion) == 0) :
    ciExpansion = [1.]

hffilename = "hf.txt"

hffile = open(hffilename, "r")
lines = hffile.readlines()
norbs = len(lines)
hforbs = np.zeros((norbs*len(lines[0].split()),))  
for i in range(len(lines)):
    linesp = lines[i].split();
    for j in range(len(linesp)):
        if (j < norbs ):
            hforbs[i*norbs+j] = float(linesp[j])
        else:
            hforbs[norbs*norbs+i*norbs+j-norbs] = float(linesp[j])



numCPS = numVars
numVars += len(ciExpansion) + hforbs.shape[0]
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
wrt[numCPS : numCPS+len(ciExpansion)] = np.asarray(ciExpansion)

wrt[numCPS+len(ciExpansion) : ] = hforbs
civars = len(ciExpansion)
 
if (Restart):
    wrt  = np.fromfile("params.bin", dtype = "float64")
    emin = np.fromfile("emin.bin", dtype = "float64")[0]


if (doHessian):
    for i in range(500):
        grad = d_loss_wrt_pars(wrt)
        Hessian = np.fromfile("hessian.bin", dtype="float64")
        Smatrix = np.fromfile("smatrix.bin", dtype="float64")
        Hessian.shape = (numVars+1, numVars+1)
        Smatrix.shape = (numVars+1, numVars+1)

        Hessian[1:, 1:] += 0.01*np.eye(numVars)

        #make the tangent space orthogonal to the wavefunction
        Uo = 0.* Smatrix
        Uo[0,0] = 1.0;
        for i in range(numVars):
            Uo[0, i+1] = -Smatrix[0, i+1]
            Uo[i+1, i+1] = 1.0

        Smatrix = reduce(np.dot, (Uo.T, Smatrix, Uo))
        Hessian = reduce(np.dot, (Uo.T, Hessian, Uo))

        [ds, vs] = np.linalg.eigh(Smatrix)

        cols = []
        for i in range(numVars+1):
            if (abs(ds[i]) > 1.e-8):
                cols.append(i)
        
        U = np.zeros((numVars+1, len(cols)))
        for i in range(len(cols)):
            U[:,i] = vs[:,cols[i]]/ds[cols[i]]**0.5

        Hessian_prime = reduce(np.dot, (U.T, Hessian, U))
        [dc, dv] = np.linalg.eig(Hessian_prime)
        index = [np.argmin(dc.real)]
        print "Expected energy in next step       : ", dc[index].real
        print "Number of total/nonredundant pramas: ", numVars+1, len(cols)
        sys.stdout.flush()
        update = np.dot(U, dv[:,index].real)
        dw = update[1:]/update[0]
        dw.shape = wrt.shape
        wrt += dw

else :        
    #wrt = np.fromfile("params.bin", dtype="float64")
    #opt = climin.GradientDescent(wrt, d_loss_wrt_pars, step_rate=0.01, momentum=.95)
    #opt = climin.rmsprop.RmsProp(wrt, d_loss_wrt_pars, step_rate=0.0001, decay=0.9)
    opt = climin.amsgrad.Amsgrad(wrt, d_loss_wrt_pars, step_rate=m_stepsize, decay_mom1=m_decay_mom1, decay_mom2=m_decay_mom2, momentum=0.0)

    if (Restart):
        opt.est_mom1_b = np.fromfile("moment1.bin", dtype="float64")
        opt.est_mom2_b = np.fromfile("moment2.bin", dtype="float64")

    for info in opt:
        if info['n_iter'] >= maxIter:
            break
        opt.est_mom1_b.astype("float64").tofile("moment1.bin")
        opt.est_mom2_b.astype("float64").tofile("moment2.bin")

        if (os.path.isfile("updateparams.txt")):
            f = open("updateparams.txt", 'r')
            for line in f:
                linesp = line.split();            
                if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "stepsize"):
                    m_stepsize = float(linesp[1])
                if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "decay1"):
                    m_decay_mom1 = float(linesp[1])
                if (len(linesp) != 0 and linesp[0][0] != "#" and linesp[0].lower() == "decay2"):
                    m_decay_mom2 = float(linesp[1])

                print "#updating opt-params"
                print "#stepsize   : %f"%(m_stepsize)
                print "#decay_mom1 : %f"%(m_decay_mom1)
                print "#decay_mom2 : %f"%(m_decay_mom2)
                print "#**********"

                opt.step_rate = m_stepsize
                opt.decay_mom1 = m_decay_mom1
                opt.decay_mom2 = m_decay_mom2

            f.close()
            os.remove("updateparams.txt")
