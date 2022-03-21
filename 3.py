# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
from scipy.stats import norm, t

alpha = 0.12360000000000002
beta = 0.8527399999999997
omega = .01**2 * 0.027929999999999993

#Import from csv
df = pd.read_csv('DJI2007_2016.csv')
df['Date'] = pd.to_datetime(df.Date)
# Calculate log-returns
df['LogReturn'] = np.log(df['Close']).diff()

variance = np.var(df['LogReturn'][1:len(df['LogReturn'])].to_numpy())
garch = [math.sqrt(variance), math.sqrt(variance)]
var = variance
for i in range(2,len(df['LogReturn'])):
    var = alpha*(df['LogReturn'][i-1]**2) + beta*var + omega
    garch.append(var)
df['GARCH'] = np.sqrt(garch)

K = 200
red = mcolors.to_rgb('#ff5050')
#a)
longtermVariance = omega/(1-(alpha+beta))
print('Unconditional Volatility:', math.sqrt(longtermVariance))

#b)
acfReturns = []
returns = df['LogReturn'][1:len(df['LogReturn'])].to_numpy()
meanReturn = np.mean(df['LogReturn'][1:len(df['LogReturn'])].to_numpy())
variance = np.var(df['LogReturn'][1:len(df['LogReturn'])].to_numpy())
for k in range(1,K):
    ck = []
    for t in range(len(returns)-k):
        ck.append((returns[t]-meanReturn)*(returns[t+k]-meanReturn))
    acfReturns.append((sum(ck)/len(df['LogReturn']))/variance)
fig, acf = plt.subplots()
black = mcolors.to_rgb('#000000')
p1, = acf.plot(range(1,K),acfReturns, color=black, label="Afc")
plt.title('AFC Log Returns')
plt.ylabel('Correlation')
plt.xlabel('Lag')
plt.show()


squaredReturns = df['LogReturn'][1:len(df['LogReturn'])].to_numpy()**2
acfSquaredReturns = []
meanSquaredReturns = np.mean(squaredReturns)
varianceSquaredReturns = np.var(squaredReturns)
for k in range(1,K):
    ck = []
    for t in range(len(squaredReturns)-k):
        ck.append((squaredReturns[t]-meanSquaredReturns)*(squaredReturns[t+k]-meanSquaredReturns))
    acfSquaredReturns.append((sum(ck)/len(df['LogReturn']))/varianceSquaredReturns)
fig, acf = plt.subplots()
p1, = acf.plot(range(1,K),acfSquaredReturns, color=black, label="Afc")
plt.title('AFC Squared Log Returns')
plt.ylabel('Correlation')
plt.xlabel('Lag')
plt.show()



#c)
residuals = []
for i in range(1,len(df['LogReturn'])):
    residuals.append(df['LogReturn'][i]/df['GARCH'][i])
acfResiduals = []
meanResidual = np.mean(residuals)
varianceResidual = np.var(residuals)
for k in range(1,K):
    ck = []
    for t in range(100,len(residuals)-k):
        ck.append((residuals[t]-meanResidual)*(residuals[t+k]-meanResidual))
    acfResiduals.append((sum(ck)/len(residuals))/varianceResidual)
fig, acf = plt.subplots()
p1, = acf.plot(range(1,K),acfResiduals, color=black, label="Afc")
plt.title('AFC Residuals')
plt.ylabel('Correlation')
plt.xlabel('Lag')
plt.show()


squaredResiduals = np.array(residuals)**2
acfSquareResiduals = []
meanSquareResidual = np.mean(squaredResiduals)
varianceSquareResidual = np.var(squaredResiduals)
for k in range(1,K):
    ck = []
    for t in range(100,len(squaredResiduals)-k):
        ck.append((squaredResiduals[t]-meanSquareResidual)*(squaredResiduals[t+k]-meanSquareResidual))
    acfSquareResiduals.append((sum(ck)/len(squaredResiduals))/varianceSquareResidual)
fig, acf = plt.subplots()
p1, = acf.plot(range(1,K),acfSquareResiduals, color=black, label="Afc")
plt.title('AFC Squared Residuals')
plt.ylabel('Correlation')
plt.xlabel('Lag')
plt.show()



#d)
mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 5*sigma, mu + 4*sigma, 100)
plt.plot(x, norm.pdf(x, mu, sigma), label='Standard Normal Distiubution', color=black)
plt.hist(residuals[100:-1], 100, color=red, label="Residuals", alpha=0.8, density=True)
plt.legend(loc='best')
plt.title('Resiudal- VS Normal Distrubution')
plt.show()



#e)
kurt = pd.DataFrame(residuals).kurtosis()
print(kurt[0])

residuals.sort()

red = mcolors.to_rgb('#ff5050')
black = mcolors.to_rgb('#000000')
theorethicalQ = np.arange(1,len(residuals)+1)/(len(residuals))
x = [-6,6]
y = x

theorethicalV = t.ppf(theorethicalQ, 3, loc=0, scale=1)
plt.scatter(theorethicalV,residuals, color=black)
plt.plot(x,y, color=red)
plt.title('QQ-Plot, Student T DF=3')
plt.xlabel('Theoretical Quantile')
plt.ylabel('Observed Quantile')
plt.show()

theorethicalV = t.ppf(theorethicalQ, 7, loc=0, scale=1)
plt.scatter(theorethicalV,residuals, color=black)
plt.plot(x,y, color=red)
plt.title('QQ-Plot, Student T DF=7')
plt.xlabel('Theoretical Quantile')
plt.ylabel('Observed Quantile')
plt.show()

theorethicalV = t.ppf(theorethicalQ, 8, loc=0, scale=1)
plt.scatter(theorethicalV,residuals, color=black)
plt.plot(x,y, color=red)
plt.title('QQ-Plot, Student T DF=8')
plt.xlabel('Theoretical Quantile')
plt.ylabel('Observed Quantile')
plt.show()

theorethicalV = t.ppf(theorethicalQ, 9, loc=0, scale=1)
plt.scatter(theorethicalV,residuals, color=black)
plt.plot(x,y, color=red)
plt.title('QQ-Plot, Student T DF=9')
plt.xlabel('Theoretical Quantile')
plt.ylabel('Observed Quantile')
plt.show()

theorethicalV = t.ppf(theorethicalQ, 15, loc=0, scale=1)
plt.scatter(theorethicalV,residuals, color=black)
plt.plot(x,y, color=red)
plt.title('QQ-Plot, Student T DF=15')
plt.xlabel('Theoretical Quantile')
plt.ylabel('Observed Quantile')
plt.show()

theorethicalV = norm.ppf(theorethicalQ, loc=0, scale=1)
plt.scatter(theorethicalV,residuals, color=black)
plt.plot(x,y, color=red)
plt.title('QQ-Plot, Normal Distribution')
plt.xlabel('Theoretical Quantile')
plt.ylabel('Observed Quantile')
plt.show()