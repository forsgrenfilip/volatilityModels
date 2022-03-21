# -*- coding: utf-8 -*-
def logReturn(array):
    import math
    '''input an n lenght array with prices. supports python lists, pandas columns...
    log returns is calculated for from index 0 to n-1. index o is set to 0.
    Return an n lenght array (python list) with n-1 log returns.
    '''
    logReturnArray = []
    for i in range(len(array)-1):
        logReturnArray.append(math.log(array[i]/array[i+1]))
    logReturnArray.append(0)
    return logReturnArray