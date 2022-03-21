def GARCHParameters(logReturns):
    import numpy as np
    '''
    Solves for variance in time t using GARCH(1.1).
    i.e based on log return and the variance in t-1 to t
    '''
    logReturns = logReturns*100
    variance = np.var(logReturns)
    start = [0,0,0]
    end = [1.01,1.01,1.01]
    length = 0.01
    alpha = np.arange(start[0], end[0], length)
    beta = np.arange(start[1], end[1], length)
    omega = np.arange(start[2], end[2], length)
    logLikelyhoods = []
    indexes = []

    for l in range(4):
        if l > 0:
            if  0 < alpha and alpha < 1 :
                start[0] = alpha - length
                end[0] = alpha + length
            elif alpha == 0:
                start[0] = 0
                end[0] = alpha + length
            else:
                start[0] = alpha - length
                end[0] = 1
            
            if  0 < beta and beta < 1 :
                start[1] = beta - length
                end[1] = beta + length
            elif beta == 0:
                start[1] = 0
                end[1] = beta + length
            else:
                start[1] = beta - length
                end[1] = 1

            if  0 < omega and omega < 1 :
                start[2] = omega - length
                end[2] = omega + length
            elif omega == 0:
                start[2] = 0
                end[2] = omega + length
            else:
                start[2] = omega - length
                end[2] = 1
            length = length/10
            alpha = np.arange(start[0], end[0] + length, length)
            beta = np.arange(start[1], end[1] + length, length)
            omega = np.arange(start[2], end[2] + length, length)
        logLikelyhoods = []
        indexes = []
        var = variance

        for i in range(len(alpha)):
            for j in range(len(beta)):
                if alpha[i] + beta[j] < 1:
                    for k in range(len(omega)):
                        logLikelyhood = 0
                        for t in range(1,len(logReturns)):
                            var = omega[k] + alpha[i]*(logReturns[t-1]**2) + beta[j]*var + 0.000000000000000000000000001
                            term = np.log(var) + (logReturns[t]**2)/var
                            logLikelyhood -= term
                        logLikelyhoods.append(logLikelyhood)
                        indexes.append([i,j,k])
        
        alpha = alpha[indexes[logLikelyhoods.index(max(logLikelyhoods))][0]]
        beta = beta[indexes[logLikelyhoods.index(max(logLikelyhoods))][1]]
        omega = omega[indexes[logLikelyhoods.index(max(logLikelyhoods))][2]]
    
    omega = omega/10000

    return alpha, beta, omega