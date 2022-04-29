import pdb
import itertools

orbitalOrder=[9, 5, 16, 2, 1, 18, 19, 0, 7, 15, 4, 14, 3, 12, 11, 17, 8, 10, 13, 6]

norbs = 20
f = open("twosite.txt", 'w')

l = []
for i in range(norbs):
    l.append(i)

combin = list(itertools.combinations(l, 2))

for t in combin:
    if ( abs(orbitalOrder[t[0]] - orbitalOrder[t[1]]) <= 1):
        f.write("%d %d \n"%( t[0], t[1]))

f.close()
