# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from MA import MA
from EWMA import EWMA

## Import from csv
df = pd.read_csv('DJI2017_2020.csv')
df['Date'] = pd.to_datetime(df.Date)
df['LogReturn'] = np.log(df['Close']).diff()

## Calculate Volatility
N = 60

ma = MA(df['LogReturn'],N)
df['VMA'] = ma

ewma = EWMA(df['LogReturn'],N,0.96)
df['VEWMA'] = ewma

## Plot
fig, price = plt.subplots()
yVol = price.twinx()
price.set_ylim(0, 31000)
yVol.set_ylim(0, 0.15)
price.set_ylabel("Price")
yVol.set_ylabel("Volatility")
price.set_xlabel("Date")

red = mcolors.to_rgb('#ff5050')
darkRed = mcolors.to_rgb('#800000')
blue = mcolors.to_rgb('#000066')

p1, = price.plot(df['Date'],df['Close'], color=blue, label="Price")
p2, = yVol.plot(df['Date'][N:len(df['VMA'])],df['VMA'][N:len(df['VMA'])], color=red, label="MA")
p3, = yVol.plot(df['Date'][N:len(df['VEWMA'])],df['VEWMA'][N:len(df['VEWMA'])], color=darkRed, label="EWMA")


lns = [p1, p2, p3]
price.legend(handles=lns, loc='best')

price.yaxis.label.set_color(blue)
yVol.yaxis.label.set_color(darkRed)

plt.show()