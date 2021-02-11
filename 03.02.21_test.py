# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:45:35 2021

@author: ahadj
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
import copy
from scipy.optimize import curve_fit
import pandas as pd
__import__=('27.01.21_code')
#%%studying density vs average velocity
xdim=64
it=20
density=np.arange(0.2,1.2,0.1)
vc_allmean1=[]
vc_allstd1=[]
for d in density:
    print(d)
    vcratio_all=[]
    for i in range(it):
        nodes=int(xdim**2*d)
        smltn0102r=RunState(35,nodes,64,64,5,1,0.3,5,100)
        minloc=min([i for i in range(len(smltn0102r[1])) if smltn0102r[1][i][0]>xdim-1])#finds 
        for t in range(len(smltn0102r[0])):
            if set(smltn0102r[0][t][minloc:])!={0}:
                vc=xdim/t
                vcratio_all.append(5/vc)
                #print(t)
                break
    if len(vcratio_all)==0:
        vc_allmean1.append(0)
        vc_allstd1.append(0)
    else:
        vc_allmean1.append(np.mean(vcratio_all))
        vc_allstd1.append(np.std(vcratio_all)/np.sqrt(it))
#%%
plt.hlines(1,0.1,1.2)
plt.plot(density,vc_allmean1,'x')
plt.errorbar(density,vc_allmean1,yerr=vc_allstd1,fmt='.',capsize=5)
plt.xlabel('Density of nodes')
plt.ylabel('Radius to average conduction velocity ratio')
#plt.savefig('vcrvsdens',dpi=1000)
#plt.plot(radii,pfit(radii))
#%%
smltn0302=RunState(80,1000,32,32,5,1,0.35,3,20)
#%%
#writergif = animation.PillowWriter()
#MovieNodes(smltn0302,None).save('testcase0302.gif',writer=writergif)
MovieNodes(smltn0302,None)
#%%
vectors0302=Resultant_Vectors(smltn0302,outv=True)
pxv0302=PixelatedVectors(smltn0302,vectors0302,32,32)
#%%
writergif = animation.PillowWriter()
VectorMovie(pxv0302[0],pxv0302[1],None).save('testcasevec0302.gif',writer=writergif)
#VectorMovie(pxv0302[0],pxv0302[1],None)
#%%
xdim=32
tM=TimeLoc(pxv0302[0],pxv0302[1],smltn0302)
print(tM)
#VectorMovie(pxvM27[0],pxvM27[1],tM)
#%%
locall=[]
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn0302[0])-5):
        print(t)
        q_all=[]
        for i in range(32**2):
            q=focal_quality_indicator(pxv0302[0],pxv0302[1],pxv0302[1][i],t,2,5)
            q_all.append(q)
        loci=q_all.index(max(q_all))
        locM=[loci%xdim,loci//xdim]
        locall.append(locM)
#%%
matrix=np.zeros([32,32])
for i in locall:
    matrix[i[1]][i[0]]+=1
fig=plt.figure()
s=plt.imshow(matrix,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
fig.colorbar(s)
plt.savefig('dotmap',dpi=1000)