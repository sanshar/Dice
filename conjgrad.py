import numpy as np
import math

#Solves Ax = b for x, where A is a real,symmetric, positive-definite matrix
#for more infomration about the conjugate gradient algorithm, wikipedia has an excellent write up

def conjgrad(H,b,xguess):
	x0 = xguess
	r0 = b - np.dot(H,x0)
	p0 = r0
	dotr0 = np.dot(r0.T,r0)
	for i in range(10):
		Ap = np.dot(H,p0)
		pAp = np.dot(p0.T,Ap)
		a = dotr0/pAp
		x1 = x0 + a*p0
		r1 = r0 - a*Ap
		dotr1 = np.dot(r1.T,r1)
		b = dotr1/dotr0
		p1 = r1 + b*p0

		x0 = x1
		r0 = r1
		p0 = p1
		dotr0 = dotr1
	return x1
