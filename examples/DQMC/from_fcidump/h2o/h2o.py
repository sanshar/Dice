import sys, os
import numpy
import numpy as np
import from_fcidump

# these need to be provided
dice_binary = "/projects/joku8258/software/Dice/Dice"
vmc_root = "/projects/joku8258/QMC/VMC/"

norb_frozen = 1
nelec = 10
norb_act = 6 
nelec_act = 8
norb_core = (nelec-nelec_act)//2 -norb_frozen # mc.ncore  

################################   

from_fcidump.prepAFQMC_fromFCIDUMP(seed = 89649,left='rhf',norb_frozen=norb_frozen,nblocks=1000,fcidump="FCIDUMP",fname='afqmc.json')
#
############################################################################
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"

print("Starting AFQMC / MF calculation", flush=True)
command = f'''
              mpirun {afqmc_binary} afqmc.json > afqmc.out;
              mv samples.dat samples_afqmc.dat
              python {blocking_script} samples_afqmc.dat 50 > blocking_afqmc.out;
              cat blocking_afqmc.out;
           '''
os.system(command)
print("Finished AFQMC / MF calculation")                                                             
###############################################################################                        

def prepDice():  #Example input file for dice. Change it according to the system
	dice = '''
#system
nocc 8		
0 2 4 6 1 3 5 7 
end
orbitals ./FCIDUMP
nroots 1
pointGroup c2v

#variational
schedule
0   0.001
end
davidsonTol 5e-05
dE 1e-06
maxiter 6	
#pt
nPTiter 0
epsilon2 1e-07
epsilon2Large 1000
targetError 0.0001
sampleN 200

#misc
io

writebestdeterminants 100000
'''
	f = open("dice.dat","w")
	f.write(dice)
	f.close()	

prepDice()
print("Running Dice")
command = f"mpirun {dice_binary} dice.dat > dice.out;rm *.bkp"	
os.system(command)
print("Finished Dice calculation")
from_fcidump.prepAFQMC_fromFCIDUMP(seed = 89649,left = 'multislater',norb_core=norb_core,norb_frozen=norb_frozen,nblocks=1000,ndets=100,fcidump="FCIDUMP",fname='afqmc_multislater.json')

#The determinants are present in the folder as dets.bin (nroot=0) or dets_{nroot}.bin (nroot>0)
############################################################################                          
afqmc_binary = vmc_root + "/bin/DQMC"
blocking_script = vmc_root + "/scripts/blocking.py"                                                   

print("Starting AFQMC / HCI calculation", flush=True)                                                 
command = f'''
              mpirun {afqmc_binary} afqmc_multislater.json > afqmc_multislater.out;       
              mv samples.dat samples_multislater.dat
              python {blocking_script} samples_multislater.dat 50 > blocking_multislater.out;         
              cat blocking_multislater.out;
           '''
os.system(command)
print("Finished AFQMC / HCI calculation")
################################################################################


