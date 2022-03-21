# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import math
import random

## Settings
iterations = 10000
cutOff = pd.to_datetime('2017-01-01')


# GARCH Parameters (estimation window 2007-2016)
alpha = 0.12360000000000002
beta = 0.8527399999999997
omega = .01**2 * 0.027929999999999993


# VaR Parameter
VaRAlpha = 0.01

## Import Data
df = pd.read_csv('DJI2007_2020.csv')
df['Date'] = pd.to_datetime(df.Date)
df['LogReturn'] = np.log(df['Close']).diff()

historicLogReturns = df['LogReturn'][df['Date'] < '2017-01-01'][1:-1].to_numpy()
period = len(df['LogReturn'][df['Date'] > cutOff])

# Generate Residuals from Historical Data
variance = np.var(historicLogReturns)
variances = [variance]
var = variance
for i in range(1,len(historicLogReturns)):
    var = alpha*(historicLogReturns[i-1]**2) + beta*var + omega
    variances.append(var)

residuals = []
for i in range(len(historicLogReturns)):
    residuals.append(historicLogReturns[i]/np.sqrt(variances[i]))

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 4*sigma, 100)
plt.plot(x, norm.pdf(x, mu, sigma), label='Standard Normal Distiubution')
plt.hist(residuals, 100, label="Residuals", alpha=0.8, density=True)
plt.legend(loc='best')
plt.title('Resiudal- VS Normal Distrubution')
plt.show()

futureLogReturns = df['LogReturn'][df['Date'] > '2017-01-01'].to_numpy()

# Generate Future residuals
variance = np.var(futureLogReturns)
variances = [variance]
var = variance
for i in range(1,len(futureLogReturns)):
    var = alpha*(futureLogReturns[i-1]**2) + beta*var + omega
    variances.append(var)

futResiduals = []
for i in range(len(futureLogReturns)):
    futResiduals.append(futureLogReturns[i]/np.sqrt(variances[i]))
futResiduals.sort()

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 4*sigma, 100)
plt.plot(x, norm.pdf(x, mu, sigma), label='Standard Normal Distiubution')
plt.hist(futResiduals, 100, label="Residuals", alpha=0.8, density=True)
plt.legend(loc='best')
plt.title('Resiudal- VS Normal Distrubution')
plt.show()


## FHS; Simulate LogReturns with GARCH (volatility) and Randomly Drawns Residual (noice)

FHSsimulatedReturns = np.zeros(iterations)
FHSVaR = np.zeros(period)
logReturn = historicLogReturns[-1]
var = np.var(historicLogReturns)
for t in range(period):
    var = alpha*logReturn**2 + beta*var + omega
    for i in range(iterations):
        z = residuals[random.randint(0,len(residuals)-1)]
        logReturn = math.sqrt(var)*-z
        FHSsimulatedReturns[i] = logReturn

    FHSsimulatedReturns.sort()
    FHSVaR[t] = np.quantile(FHSsimulatedReturns, 1-VaRAlpha)

    logReturn = futureLogReturns[t]

violations = 0
for i in range(len(FHSVaR)):
    if FHSVaR[i] < -futureLogReturns[i]:
        violations += 1

violationsPrc = violations/len(FHSVaR)
violationRatio = violationsPrc/VaRAlpha

print([violations,violationRatio])

## Simulate LogReturns with GARCH (volatility) and Normal Random Variable (noice)
mu, sigma = 0, 1
GARCHsimulatedReturns = np.zeros(iterations)
GARCHVaR = np.zeros(period)
logReturn = historicLogReturns[-1]
var = variances[-1]
for t in range(period):
    var = alpha*logReturn**2 + beta*var + omega
    for i in range(iterations):
        z = np.random.normal(mu, sigma)
        logReturn = math.sqrt(var)*-z
        GARCHsimulatedReturns[i] = logReturn

    GARCHsimulatedReturns.sort()
    GARCHVaR[t] = np.quantile(GARCHsimulatedReturns, 1-VaRAlpha)

    logReturn = futureLogReturns[t]

violations = 0
for i in range(len(GARCHVaR)):
    if GARCHVaR[i] < -futureLogReturns[i]:
        violations += 1

violationsPrc = violations/len(GARCHVaR)
violationRatio = violationsPrc/VaRAlpha

print([violations,violationRatio])


VaRAlpha = 0.01
cutOff = pd.to_datetime('2017-01-01')

#Import from csv
df = pd.read_csv('DJI2013_2020.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['LogReturn'] = np.log(df['Close']).diff()

period = len(df['LogReturn'][df['Date'] > cutOff])
futureReturns = df['LogReturn'][df['Date'] > cutOff].values.tolist()

windowSize = 1000

rollingReturnWindow = df['LogReturn'][df['Date'] < cutOff][-(windowSize+1):-1].values
VaR = np.zeros(period)
for t in range(period):
    simulatedReturns = np.zeros(iterations)
    for i in range(iterations):
        pick = int(np.floor(np.random.rand(1)*windowSize)[0])
        simulatedReturns[i] = -rollingReturnWindow[pick]
    simulatedReturns.sort()
    VaR[t] = np.quantile(simulatedReturns, 1-VaRAlpha)
    rollingReturnWindow = np.append(rollingReturnWindow,futureReturns[t])
    rollingReturnWindow = np.delete(rollingReturnWindow, 0)

violations = 0
for i in range(len(VaR)):
    if VaR[i] < -futureReturns[i]:
        violations += 1

violationsPrc = violations/len(VaR)
violationRatio = violationsPrc/VaRAlpha

print([violations,violationRatio])

windowSize = 500

rollingReturnWindow = df['LogReturn'][df['Date'] < cutOff][-(windowSize+1):-1].values
VaR1 = np.zeros(period)
for t in range(period):
    simulatedReturns = np.zeros(iterations)
    for i in range(iterations):
        pick = int(np.floor(np.random.rand(1)*windowSize)[0])
        simulatedReturns[i] = -rollingReturnWindow[pick]
    simulatedReturns.sort()
    VaR1[t] = np.quantile(simulatedReturns, 1-VaRAlpha)
    rollingReturnWindow = np.append(rollingReturnWindow,futureReturns[t])
    rollingReturnWindow = np.delete(rollingReturnWindow, 0)

violations = 0
for i in range(len(VaR1)):
    if VaR1[i] < -futureReturns[i]:
        violations += 1

violationsPrc = violations/len(VaR1)
violationRatio = violationsPrc/VaRAlpha

print([violations,violationRatio])
plt.plot(df['Date'][df['Date'] > cutOff], -df['LogReturn'][df['Date'] > cutOff], label='Actual Returns Post 2017')
plt.plot(df['Date'][df['Date'] > cutOff],FHSVaR, label='FHS VaR')
plt.plot(df['Date'][df['Date'] > cutOff],GARCHVaR, label='GARCH VaR')
plt.plot(df['Date'][df['Date'] > cutOff],VaR, label='SHS VaR Window:1000')
plt.plot(df['Date'][df['Date'] > cutOff],VaR1, label='SHS VaR Window:500')
plt.title('Simple Historical Simulation Estimation of VaR')
plt.legend(loc='best')

plt.show()






