# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 19:37:34 2023

@author: echon
"""

import statsmodels.api as sm
import pandas as pd
import numpy as np

# reading data from the csv
data = pd.read_csv('toyDataSet.csv')

# defining the variables 

x1 = data['x1'].tolist()
y = data['y'].tolist()
x2 = data['x2'].tolist()

A = np.array([x1,x2]).T 
A.shape
A = sm.add_constant(A)
A.shape

model = sm.OLS(y,A )
results = model.fit()

results.params
print (results.summary())

import matplotlib.pyplot as plt
ypred = []
for i in range (5):
    ypred.append(.1 + .82*x1[i] + 1.26*x2[i])
    
   # now do a second order fit 

plt.plot(ypred, y, 'o')
Axis = np.linspace(0, 7)
plt.plot(Axis, Axis)
plt.xlabel('x')
plt.ylabel('y')
plt.title('regression')
plt.legend()
plt.grid()
plt.show()

# use list comprehension
x3 = [x*x for x in x2]

A = np.array([x1,x2, x3]).T 
A.shape
A = sm.add_constant(A)
A.shape

model = sm.OLS(y,A )
results = model.fit()

results.params
print (results.summary())

ypred2 = []
for i in range (5):
    ypred2.append(.1 + .77*x1 [i]+ 1.39*x2[i] + -.05*x3[i])

plt.plot(ypred2, y , 'x')
plt.plot(ypred, y, 'o', color = "red")
