# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 20:48:03 2021

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
__import__=('27.01.21_code')
#%%
res=np.arange(6,64,2)
d_all=[]
for it in range(20):
    print(it)
    vcond=3
    tau=5
    smltnM27=RunState(80,2000,64,64,5,1,0.26,5,20)
    vectorsM27=Resultant_Vectors(smltnM27,outv=True)
    pxvM27=PixelatedVectors(smltnM27,vectorsM27,64,64)
    tM=TimeLoc(pxvM27[0],pxvM27[1],smltnM27)
    #VectorMovie(pxvM27[0],pxvM27[1],tM)
#    if tM==0:
#        locM=[0,0]
#    else:
#        q_all=[]
#        for i in range(64**2):
#            q=focal_quality_indicator(pxvM27[0],pxvM27[1],pxvM27[1][i],tM,vcond,tau)
#            q_all.append(q)
#        loci=q_all.index(max(q_all))
#        locM=[loci%64,loci//64]
    rint=random.randint(0,len(pxvM27[0][0])-1)
    locM=pxvM27[1][rint]
    
    d_r=[]
    for r in res:
        b=64/r
        pxvM27r=PixelatedVectors(smltnM27,vectorsM27,r,r)
        tMr=TimeLoc(pxvM27r[0],pxvM27r[1],smltnM27)
        #VectorMovie(pxvM27r[0],pxvM27r[1],tMr)
#        if tMr==0:
#            locr=[0,0]
#        else:
#            q_allr=[]      
#            for i in range(r**2):           
#                qr=focal_quality_indicator(pxvM27r[0],pxvM27r[1],pxvM27r[1][i],tMr,vcond/b,tau) 
#                q_allr.append(qr)
#            locir=q_allr.index(max(q_allr))
#            locr=[locir%r,locir//r]
        rintr=random.randint(0,len(pxvM27r[0][0])-1)
        locr=pxvM27r[1][rintr]
        
        locMa=np.array(locM)//b
        disti=np.sqrt((locMa[0]-locr[0])**2+(locMa[1]-locr[1])**2)
        d_r.append(disti)
    
    d_all.append(d_r)
    
#%%
#np.savetxt('spacediffrand2701.txt',d_all)
#%%
spacediff=[]#list with space difference for each resolution
for r in range(len(res)):
    spacediffL=[]#list for space diff for a single resolution
    for it in range(len(d_all)):
        spacediffL.append(d_all[it][r])
    spacediff.append(spacediffL)
#%%
d_mean=np.array([np.mean(i) for i in spacediff])
d_sd=np.array([np.std(i) for i in spacediff])/np.sqrt(20)
vcall=np.array([vcond*i/64 for i in res])#conduction velocity for all the vectors
pathl=5*vcall
dtow=d_mean/pathl#distance to wavelength ratio
dtowsd=d_sd/pathl
xaxis=np.array([64]*len(res))/res#bcoefficient
#%%
plt.hlines(1,1,12)
plt.plot(xaxis,dtow,'x')
plt.errorbar(xaxis,dtow,yerr=dtowsd,fmt='.',capsize=5)
plt.xlabel('b coefficient')
plt.ylabel('Distance to wavelength ratio')  
#plt.savefig('dtowrandom2701',dpi=1000)
#%%
plt.plot(res,dtow,'x')
plt.errorbar(res,dtow,yerr=dtowsd,fmt='.',capsize=5)
plt.xlabel('Resolution')
plt.ylabel('Distance to wavelength ratio')  
#plt.savefig('dtowrandomres2701',dpi=1000)


        