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

residuals = []
for i in range(1,len(df['LogReturn'])):
    residuals.append(df['LogReturn'][i]/df['GARCH'][i])

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