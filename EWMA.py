# -*- coding: utf-8 -*-
def EWMA(array,N,lambd):
    import math
    '''input an n lenght array with prices. supports python lists, pandas columns...
    N-day Exponantially Weighted Moving average is calculated for from t=N+1 to t=n.
    Return an n lenght array (python list) with N-day moving averages and n-N zeros.
    '''
    EWMAArray = []
    for i in range(N):
        EWMAArray.append(0)
    for i in range(N,len(array)):
        window = []
        for j in range(1,N+1):
            window.append((lambd**j) * (array[i-j]**2))
        EWMAArray.append(math.sqrt(((1-lambd)/(lambd*(1-lambd**N))) * sum(window)))

    return EWMAArray