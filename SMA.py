# -*- coding: utf-8 -*-
def SMA(array,N):
    '''input an n lenght array with prices. supports python lists, pandas columns...
    N-day Simple Moving average is calculated for from t=N+1 to t=n.
    Return an n lenght array (python list) with N-day moving averages and n-N zeros.
    '''
    smaArray = []
    for i in range(len(array)-N):
        sma = sum(array[i:N+i])/N
        smaArray.append(sma)
    for i in range(N):
        smaArray.append(0)
    return smaArray