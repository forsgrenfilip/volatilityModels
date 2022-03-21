# -*- coding: utf-8 -*-
def MA(array,N):
    import math
    '''input an n lenght array with prices. supports python lists, pandas columns...
    N-day Simple Moving average is calculated for from t=N+1 to t=n.
    Return an n lenght array (python list) with N-day moving averages and n-N zeros.
    '''
    maArray = []
    for i in range(N):
        maArray.append(0)
    for i in range(N,len(array)):
        maArray.append(math.sqrt(sum(array[i-N:i]**2)/N))
    return maArray