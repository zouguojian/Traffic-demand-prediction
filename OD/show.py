# -- coding: utf-8 --

import  matplotlib.pyplot as plt
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:09:06 2018

@author: butany
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

mae_45_15=[0.837,0.794,0.806,0.806,0.848]
mae_120_45=[0.875,0.824,0.832,0.864,0.935]
mae_120_90=[0.880,0.847,0.939, 0.987,1.005]

rsme_45_15=[1.431,1.409,1.417,1.433,1.449]
rsme_120_45=[1.470,1.435,1.499,1.536,1.620]
rsme_120_90=[1.540,1.488,1.657,1.709,1.712]

r_45_15=[0.932,0.936,0.935,0.930,0.930]
r_120_45=[0.931,0.934,0.932,0.925,0.916]
r_120_90=[0.924,0.927,0.918,0.907,0.907]

plt.figure()
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}
plt.ylabel('Loss(ug/m3)',font2)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 6,
}

plt.subplot(1,3,1)

plt.plot(range(1,6,1),mae_45_15,marker='o',color='orange',linestyle='--', linewidth=1,label='45-15 min')
plt.plot(range(1,6,1), mae_120_45,marker='s', color='#0cdc73',linestyle='-.',linewidth=1,label='120-45 min')
plt.plot(range(1,6,1), mae_120_90,marker='p', color='#f504c9',linestyle='-',linewidth=1,label='120-90 min')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.ylabel('MAE value',font2)
plt.xlabel('layer',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,2)
plt.plot(range(1,6,1),rsme_45_15,marker='o',color='#d0c101',linestyle='--', linewidth=1,label='45-15 min')
plt.plot(range(1,6,1), rsme_120_45,marker='s', color='#ff5b00',linestyle='-.',linewidth=1,label='120-45 min')
plt.plot(range(1,6,1), rsme_120_90,marker='p', color='#a8a495',linestyle='-',linewidth=1,label='120-90 min')
plt.legend(loc='upper left',prop=font1)
plt.grid(axis='y')
plt.ylabel('RMSE value',font2)
plt.xlabel('layer',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.subplot(1,3,3)
plt.plot(range(1,6,1),r_45_15,marker='o',color='#ffdf22',linestyle='--', linewidth=1,label='45-15 min')
plt.plot(range(1,6,1), r_120_45,marker='s', color='#82cafc',linestyle='-.',linewidth=1,label='120-45 min')
plt.plot(range(1,6,1), r_120_90,marker='p', color='#a55af4',linestyle='-',linewidth=1,label='120-90 min')
plt.ylabel('R$^2$ value',font2)
plt.xlabel('layer',font2)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(loc='upper right',prop=font1)
plt.grid(axis='y')


plt.show()