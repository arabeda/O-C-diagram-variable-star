import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmin, find_peaks
"""Minima fitting for a variable star"""
BJD = 2457000 #barycentric julian day
"""t_i is a 'i' time of observation and t_0 is the first data point, P is a star peroid from publication"""
P = 0.194934
f = open('data.txt')
x, y=[], []
for l in f.readlines():
    tmp = l.split('  ')
    x.append(float(tmp[1]))
    y.append(float(tmp[-1]))
m_0 = 1683.51189099
print(m_0)

d = pd.read_csv('phasedata.tsv', sep='\t', header=None)
d= d.values
d = d[:,:3]

ph = d[:,0]
t = d[:,1]
mag = d[:,2]
E = -np.floor((m_0-t)/P)
E = np.sort(np.array(list(set(E))))
C = m_0 +E*P
minima = find_peaks(-mag, width=60)
O = t[minima[0]]
plt.plot(t,mag)
plt.show()
O_epoka = -np.floor((m_0-O)/P)

O_final = []
C_final = []
E_final = []
for i,e in enumerate(E):
    C_index = np.argwhere(O_epoka==e)
    if len(C_index):
        O_final.append(O[C_index[0][0]])
        C_final.append(C[i])
        E_final.append(e)

O_final = np.array(O_final)
C_final = np.array(C_final)
E_final = np.array(E_final)



plt.scatter(t,mag, s=0.1 )
plt.scatter(t[minima[0]], mag[minima[0].astype(int)], s=0.2, c='r')
plt.scatter(t[minima[0]], mag[minima[0].astype(int)], s=0.2, c='r')
plt.show()

#plt.plot()
plt.scatter(E_final, O_final-C_final, s=0.1)
plt.show()
