# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 12:53:09 2021

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
#%%need to count conduction velocity with corrected method
smltn3001=RunState(40,2000,64,64,5,1,0.28,5,100)
xdim=64
#%%
MovieNodes(smltn3001,None)
#%%
minloc=min([i for i in range(len(smltn3001[1])) if smltn3001[1][i][0]>xdim-1])#finds 
#location in coordinates of node at the end of the grid
#smltns=set(smltn3001[0][1][minloc:])
for t in range(len(smltn3001[0])):
    if set(smltn3001[0][t][minloc:])!={0}:
        print(t)
        break
#%%automating the above
xdim=64
it=100
vc_all=[]
#t_all=[]
for i in range(it):
    smltn3001r=RunState(40,2000,64,64,5,1,0.3,5,100)
    minloc=min([i for i in range(len(smltn3001r[1])) if smltn3001r[1][i][0]>xdim-1])#finds 
    for t in range(len(smltn3001r[0])):
        if set(smltn3001r[0][t][minloc:])!={0}:
            #print(minloc)
            vc=xdim/t
            vc_all.append(vc)
            #t_all.append(t)
            break
#%%mean cond velocity and stand deviation
vc_m=np.mean(vc_all)
vc_std=np.std(vc_all)/np.sqrt(it)
#%%
res=np.arange(6,64,2)
d_all=[]
for it in range(111):
    print(it)
    vcond=4.75
    tau=5
    smltnM27=RunState(80,2000,64,64,5,1,0.3,5,20)
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
#np.savetxt('spacediffrand0102new.txt',d_all)
#%%preparing all the data
res=np.arange(6,64,2)
vcond=4.75
d_allold=np.loadtxt('spacediffrand3101old.txt')
d_allnew=np.loadtxt('spacediffrand3101new.txt')
d_all=np.concatenate([d_allold,d_allnew])
d_allr=np.loadtxt('spacediffrand0102new.txt')
#%%for data
spacediff=[]#list with space difference for each resolution
for r in range(len(res)):
    spacediffL=[]#list for space diff for a single resolution
    for it in range(len(d_all)):
        spacediffL.append(d_all[it][r])
    spacediff.append(spacediffL)
#%%
d_mean=np.array([np.mean(i) for i in spacediff])
d_sd=np.array([np.std(i) for i in spacediff])/np.sqrt(len(d_all))
vcall=np.array([vcond*i/64 for i in res])#conduction velocity for all the vectors
pathl=5*vcall
dtow=d_mean/pathl#distance to wavelength ratio
dtowsd=d_sd/pathl
xaxis=np.array([64]*len(res))/res#bcoefficient
#%%for random data
spacediff=[]#list with space difference for each resolution
for r in range(len(res)):
    spacediffL=[]#list for space diff for a single resolution
    for it in range(len(d_allr)):
        spacediffL.append(d_allr[it][r])
    spacediff.append(spacediffL)
#%%
d_mean=np.array([np.mean(i) for i in spacediff])
d_sd=np.array([np.std(i) for i in spacediff])/np.sqrt(len(d_all))
vcall=np.array([vcond*i/64 for i in res])#conduction velocity for all the vectors
pathl=5*vcall
dtowr=d_mean/pathl#distance to wavelength ratio
dtowsdr=d_sd/pathl
xaxis=np.array([64]*len(res))/res#bcoefficient
#%%
plt.hlines(1,1,12)
plt.plot(xaxis,dtow,'x',label='Dot Product')
plt.errorbar(xaxis,dtow,yerr=dtowsd,fmt='.',capsize=5)
plt.plot(xaxis,dtowr,'o',label='Random Vectors')
plt.errorbar(xaxis,dtowr,yerr=dtowsdr,fmt='.',capsize=5)
plt.xlabel('b coefficient')
plt.ylabel('Cell Distance to wavelength ratio')  
plt.legend()
#plt.savefig('dtowvsb0202',dpi=1000)
#%%
plt.hlines(1,1,64)
plt.plot(res,dtow,'x',label='Dot Product')
plt.errorbar(res,dtow,yerr=dtowsd,fmt='.',capsize=5)
plt.plot(res,dtowr,'o',label='Random Vectors')
plt.errorbar(res,dtowr,yerr=dtowsdr,fmt='.',capsize=5)
plt.xlabel('Resolution')
plt.ylabel('Cell Distance to wavelength ratio')  
plt.legend()
#plt.savefig('dtowvsres0202',dpi=1000)
#%%testing the model above
smltn3001at=RunState(80,2000,64,64,5,1,0.3,5,20)
vectors3001at=Resultant_Vectors(smltn3001at,outv=True)
#%%
r=40
pxv30at=PixelatedVectors(smltn3001at,vectors3001at,r,r)
tat=TimeLoc(pxv30at[0],pxv30at[1],smltn3001at)
#%%
VectorMovie(pxv30at[0],pxv30at[1],tat+2)
#%%
tau=5
q_all=[]
for i in range(r**2):
    q=focal_quality_indicator(pxv30at[0],pxv30at[1],pxv30at[1][i],tat,vcond/(64/r),tau)
    q_all.append(q)
loci=q_all.index(max(q_all))
locM=[loci%r,loci//r]
print(locM)
