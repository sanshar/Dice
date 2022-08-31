# First do a variational dice calculation that generates a bkp file with the wave function
mpirun $Dice_binary dice.dat > dice.out

# Then calculate the greens function using
mpirun $GreensFunction_binary green.dat > green.out

This [colab notebook](https://colab.research.google.com/drive/18CNfvcmBHehW4ESBZSaN5-_YvO7QG9PE?usp=sharing) has python code for exact calculations.

