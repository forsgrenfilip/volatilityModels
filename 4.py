# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math

iterations = 10000
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
    VaR[t] = simulatedReturns[math.floor(((len(simulatedReturns)-1)*(1-VaRAlpha)))]
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
    VaR1[t] = simulatedReturns[math.floor(((len(simulatedReturns)-1)*(1-VaRAlpha)))]
    rollingReturnWindow = np.append(rollingReturnWindow,futureReturns[t])
    rollingReturnWindow = np.delete(rollingReturnWindow, 0)

violations = 0
for i in range(len(VaR1)):
    if VaR1[i] < -futureReturns[i]:
        violations += 1

violationsPrc = violations/len(VaR1)
violationRatio = violationsPrc/VaRAlpha

print([violations,violationRatio])

red = mcolors.to_rgb('#ff5050')

plt.plot(df['Date'][df['Date'] > cutOff], -df['LogReturn'][df['Date'] > cutOff], label='Actual Returns Post 2017')
plt.plot(df['Date'][df['Date'] > cutOff],VaR, label='SHS VaR Window:1000')
plt.plot(df['Date'][df['Date'] > cutOff],VaR1, label='SHS VaR Window:500')
plt.title('Simple Historical Simulation Estimation of VaR')
plt.legend(loc='best')
plt.show()

