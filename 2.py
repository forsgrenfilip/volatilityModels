# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from MA import MA
from EWMA import EWMA

alpha = 0.12360000000000002
beta = 0.8527399999999997
omega = .01**2 * 0.027929999999999993

#Import from csv
df = pd.read_csv('DJI2017_2020.csv')
df['Date'] = pd.to_datetime(df.Date)
# Calculate log-returns
df['LogReturn'] = np.log(df['Close']).diff()

# Calculate Volatility
N = 60
ma = MA(df['LogReturn'],N)
df['VMA'] = ma

ewma = EWMA(df['LogReturn'],N,0.96)
df['VEWMA'] = ewma

variance = np.var(df['LogReturn'][1:len(df['LogReturn'])].to_numpy())
garch = [variance, variance]
var = variance
for i in range(2,len(df['LogReturn'])):
    var = alpha*(df['LogReturn'][i-1]**2) + beta*var + omega
    garch.append(var)
df['GARCH'] = np.sqrt(garch)

# Plot Formatting:
fig, price = plt.subplots()
yVol = price.twinx()
price.set_ylim(0, 31000)
yVol.set_ylim(0, 0.16)
price.set_ylabel("Price")
yVol.set_ylabel("Volatility")
price.set_xlabel("Date")

red = mcolors.to_rgb('#ff5050')
darkRed = mcolors.to_rgb('#800000')
blue = mcolors.to_rgb('#000066')
orange = mcolors.to_rgb('#ff9900')

p1, = price.plot(df['Date'],df['Close'], color=blue, label="Price")
p2, = yVol.plot(df['Date'][N:len(df['VMA'])],df['VMA'][N:len(df['VMA'])], color=orange, label="MA")
p3, = yVol.plot(df['Date'][N:len(df['VEWMA'])],df['VEWMA'][N:len(df['VEWMA'])], color=red, label="EWMA")
p4, = yVol.plot(df['Date'][100:len(df['Date'])],df['GARCH'][100:len(df['GARCH'])], color=darkRed, label="GARCH")

lns = [p1, p2, p3, p4]
price.legend(handles=lns, loc='best')

price.yaxis.label.set_color(blue)
yVol.yaxis.label.set_color(darkRed)
plt.title('Volatility Estimates')

plt.show()