import numpy as np
from itertools import permutations

def perm_parity(lst_in):
    '''\
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    lst = lst_in.copy()
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity

def detExpression(lst):
    n = len(lst)
    for perm in permutations(range(n)):
        pl = list(perm)
        parity = '{} '.format('+' if perm_parity(pl) == 1 else '-')
        productFS = "".join([ '{} * ' for i in range(n - 1) ]) + '{}'
        permElements = [ lst[i][pl[i]] for i in range(n) ]
        expr = parity + productFS.format(*permElements)
        print(expr)

def column_mat_det(nc):
    print('{} column{}'.format(nc, 's' if nc > 1 else ''))
    tc = [ [ f'tc(w.ciExcitations[j][0][{p}], w.ciExcitations[j][1][{t}])' for t in range(nc) ] for p in range(nc) ]
    detExpression(tc)
    print('\n')

def laplace_det(nc):
    print('{} column{}'.format(nc, 's' if nc > 1 else ''))
    for mu in range(nc):
        tc = [ [ f'walk.walker.refHelper.tc(ref.ciExcitations[i][0][{p}], ref.ciExcitations[i][1][{t}])' for t in range(nc) ] for p in range(nc) ]
        tc[mu] = [ f's(ref.ciExcitations[i][0][{mu}], ref.ciExcitations[i][1][{t}])' for t in range(nc) ]
        detExpression(tc)
    print('\n')

def row1_mat_det(nc):
    print('1 row {} column{}'.format(nc, 's' if nc > 1 else ''))
    rt = [ 'rtSlice' ]
    rtc = [ f'refHelper.rtc_b(mCre[0], ref.ciExcitations[j][1][{t}])' for t in range(nc) ]
    t = [ f'refHelper.t(ref.ciExcitations[j][0][{p}], mDes[0])' for p in range(nc)]
    tc = [ [ f'refHelper.tc(ref.ciExcitations[j][0][{p}], ref.ciExcitations[j][1][{t}])' for t in range(nc) ] for p in range(nc) ]
    m = [ rt + rtc ] + [ [t[i]] + tc[i] for i in range(nc) ]
    detExpression(m)
    print('\n')

def row2_mat_det(nc):
    print('2 rows {} column{}'.format(nc, 's' if nc > 1 else ''))
    rt = [ [ f'rtSlice({a}, {i})' for i in range(2) ] for a in range(2) ]
    rtc = [ [ f'refHelper.rtc_b(mCre[{a}], ref.ciExcitations[k][1][{t}])' for t in range(nc) ] for a in range(2) ]
    t = [ [ f'refHelper.t(ref.ciExcitations[k][0][{p}], mDes[{i}])' for i in range(2) ] for p in range(nc) ]
    tc = [ [ f'refHelper.tc(ref.ciExcitations[k][0][{p}], ref.ciExcitations[k][1][{t}])' for t in range(nc) ] for p in range(nc) ]
    m = [ rt[i] + rtc[i] for i in range(2) ] + [ t[i] + tc[i] for i in range(nc) ]
    detExpression(m)
    print('\n')

if __name__ == "__main__":
    #for i in range(4):
    #    row1_mat_det(i+1)

    #for i in range(3):
    #    row2_mat_det(i+1)
    for i in range(3):
        laplace_det(i+2)
