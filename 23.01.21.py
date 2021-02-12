# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 23:10:39 2021

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
__import__=('21.01.21_code')
#%%following sections aim to find a new method to locate the time frame, using 
#only the time averaged sd
smltn2301=RunState(80,2000,64,64,5,1,0.25,5,20)
vectors2301=Resultant_Vectors(smltn2301,outv=True,inv=True)
#%%
r=20
pxv2301=PixelatedVectors(smltn2301,vectors2301,r,r)
#%%
MovieNodes(smltn2301,17)
#VectorMovie(pxv2301[0],pxv2301[1],62)
#%%
m=3
divstd2301=[np.std(divergence(t,pxv2301[1],r,r)[0]) for t in pxv2301[0]]
divstd2301mavg=[]
for i in range(m,len(divstd2301)-m):
    divstd2301mavg.append(np.mean(divstd2301[i-m:i+m+1]))

time=np.arange(m,80-m,1)
plt.plot(time,divstd2301mavg,'o')
#%%
smltn2501=RunState(80,2000,64,64,5,1,0.25,5,20)
vectors2501=Resultant_Vectors(smltn2501,outv=True)
#%%
MovieNodes(smltn2501,None)
#%%
r=64#set the resolution u want
px2501=Pixelation(smltn2501,r,r)
pxv2501=PixelatedVectors(smltn2501,vectors2501,r,r)
#%%
t2501=np.arange(0,80,1)
plt.plot(t2501,px2501[2][1920])#plot of the time series of a cell in grid
#%%
tsn2501=[]#contains the time series of each node
for c in range(2000):
    tsn2501.append([smltn2501[0][t][c] for t in t2501])
plt.plot(t2501,tsn2501[500])#plot of time series of node
#%%testing whether 
point=tsn2501[500]
d=80
T=20
texc=[]
for t in range(len(point)):
    if point[t]>0:#to identify the first point to be excited
        ex=point[t]
        slc=(d-t)//T
        for s in range(slc+1): 
            v=point[t+ex+s*T:t+(s+1)*T-1]
            if set(v)!={0} and set(v)!=set():#checks whether there is an unexpected excitation
                for i in range(len(v)):
                    if v[i]>0:
                        print(t+ex+s*T+i)#this is the suspicious time identified
                        break
            break        
        break
#%%improving the divergence method
smltn2701=RunState(80,2000,64,64,5,1,0.20,5,20)
vectors2701=Resultant_Vectors(smltn2701,outv=True)
#%%
r=64
pxv2701=PixelatedVectors(smltn2701,vectors2701,r,r)
#%%
#MovieNodes(smltn2701,20)
VectorMovie(pxv2701[0],pxv2701[1],10)
#%%moving average
m=2
divstd2701=[np.std(divergence(t,pxv2701[1],r,r)[0]) for t in pxv2701[0]]
divstd2701mavg=[]
for i in range(m,len(divstd2701)-m):
    divstd2701mavg.append(np.mean(divstd2701[i-m:i+m+1]))

time=np.arange(m,80-m,1)
plt.plot(time,divstd2701mavg,'o')
#%%
T=20
diff=[]
for i in range(len(divstd2701mavg)-1):
    diffi=divstd2701mavg[i+1]-divstd2701mavg[i]
    diff.append(diffi)
timed=np.arange(m+1,80-m,1)
plt.plot(timed,diff,'x')
#%%time locator that only works for pixelated data
def TimeLoc(pxvectors,coord,data):
    tau=data[8][4]
    res=int(np.sqrt(len(coord)))
    divstd=[np.std(divergence(t,coord,res,res)[0]) for t in pxvectors]
    m=2
    divstdmavg=[]
    for i in range(m,len(divstd)-m):
        divstdmavg.append(np.mean(divstd[i-m:i+m+1]))    
    topdiv=max(divstdmavg[:5])
    diff=[]
    for i in range(len(divstdmavg)-1):
        diff.append(divstdmavg[i+1]-divstdmavg[i])
    topdiff=max(diff[:5])
    #plt.plot(np.arange(0,len(divstdmavg),1),divstdmavg,'x')
    for i in range(len(divstdmavg)-2):
        d1=diff[i]
        d2=diff[i+1]
        if divstdmavg[i]>topdiv and divstdmavg[i]<divstdmavg[i+1] and divstdmavg[i]<divstdmavg[i+2] and d1>topdiff and d2>topdiff:
            timef=i-1+m
            break
        if i==len(divstdmavg)-3:
            timef=0
    return timef
    
#%%testing the above function
rsltn=np.arange(10,66,2)
tdiffrla=[]
for it in range(50):
    print(it)
    smltn2701t=RunState(80,2000,64,64,5,1,0.26,5,20)
    vectors2701t=Resultant_Vectors(smltn2701t,outv=True)
    pxv2701t=PixelatedVectors(smltn2701t,vectors2701t,64,64)
    timem=TimeLoc(pxv2701t[0],pxv2701t[1],smltn2701t)
    #print(timem)
    tdiffrl=[]
    for r in rsltn:
        pxv2701tr=PixelatedVectors(smltn2701t,vectors2701t,r,r)
        timer=TimeLoc(pxv2701tr[0],pxv2701tr[1],smltn2701t)
        #print(timer)
        tdiffr=abs(timer-timem)
        tdiffrl.append(tdiffr)
    tdiffrla.append(tdiffrl)
#%%
rda=[]#list with mean time difference for each resolution
for r in range(len(rsltn)):
    rd=[]
    for it in range(len(tdiffrla)):
        rd.append(tdiffrla[it][r])
    rda.append(np.mean(rd))
#np.savetxt('timediff2601.txt',rda)
plt.plot(rsltn,rda,'x')  
#%%reviewing the focal_quality_indicator function
def focal_quality_indicator2(vectors,coord,focalpoint,timestep,vcond,tau):
    x_dim=int(np.sqrt(len(coord)))
    radius=tau*vcond
    indexf=focalpoint[1]*x_dim+focalpoint[0]
    ll=focalpoint[0]-radius
    rl=focalpoint[0]+radius
    
    vfieldnew=[[0,0] for i in range(x_dim**2)]
    for t in range(timestep-1,timestep+tau):
        for cell in range(x_dim**2):
            if cell!=indexf and ll<=coord[cell][0]<=rl:    
                if distance_checker(radius,indexf,cell,coord,x_dim) == True:
                    vfieldnew[cell][0]+=vectors[t][cell][0]
                    vfieldnew[cell][1]+=vectors[t][cell][1]
    #print(len(vfieldnew))
    def vfieldp(point,focalpoint,radius,x_dim):
        ui=point[0]-focalpoint[0]
        vi=point[1]-focalpoint[1]
        if abs(vi)<=radius:
            vi=vi
        elif abs(vi)>radius:
            if vi+radius>x_dim:
                vi=vi-x_dim
            elif vi<0:
                vi=vi+x_dim        
        mag=np.sqrt(ui**2+vi**2)        
        return [ui/mag,vi/mag]
    
    dot_summation=[]
    
    for cell in range(len(vfieldnew)):
        if vfieldnew[cell]!=[0,0]:
            #print(cell)
            mag=np.sqrt(vfieldnew[cell][0]**2+vfieldnew[cell][1]**2)
            xn=vfieldnew[cell][0]/mag
            yn=vfieldnew[cell][1]/mag
            perfp=vfieldp(coord[cell],focalpoint,radius,x_dim)
            dot_pr=xn*perfp[0]+yn*perfp[1]
            dot_summation.append(dot_pr)
    #print(dot_summation)
    quality = sum(dot_summation)/len(dot_summation)
            
    #VectorMovie([vfieldnew],coord,0)
    return quality,vfieldnew
#%%testing the above dot product function
smltn2701=RunState(80,2000,64,64,5,1,0.25,5,20)
vectors2701=Resultant_Vectors(smltn2701,outv=True)
#%%
r=20
pxv2701=PixelatedVectors(smltn2701,vectors2701,r,r)
#%%
#MovieNodes(smltn2701,14)
VectorMovie(pxv2701[0],pxv2701[1],58)
#%%
tf=TimeLoc(pxv2701[0],pxv2701[1],smltn2701)
print(tf)
#%%
s=focal_quality_indicator2(pxv2701[0],pxv2701[1],[0,0],58,3,5)
#%%
q_all=[]
for i in range(r**2):
    q=focal_quality_indicator2(pxv2701[0],pxv2701[1],pxv2701[1][i],58,3,5)
    q_all.append(q)

loci=q_all.index(max(q_all))
print(loci%r,loci//r)
#%%
q_allJ=[]
for i in range(1):
    q=focal_quality_indicator(pxv2701[0],pxv2701[1],64,64,1,1,i,55,3,5)
    q_allJ.append(q)

loci=q_all.index(max(q_all))        
print(loci%r,loci//r)
#%%testing the perfect field
def vfieldp(point,focalpoint,radius,x_dim):
    ui=point[0]-focalpoint[0]
    vi=point[1]-focalpoint[1]
    if abs(vi)<=radius:
        vi=vi
    elif abs(vi)>radius:
        if vi+radius>x_dim:
            vi=vi-x_dim
        elif vi<0:
            vi=vi+x_dim      
    mag=np.sqrt(ui**2+vi**2)  
    if mag==0:
        mag=1
    return [ui/mag,vi/mag]
#%%producing a perfect field
#currently the region of interest is correct, the rest though are wrong
vfieldl=[]
for cell in pxv2701[1]:
    vfieldl.append(vfieldp(cell,[24,51],15,64))

VectorMovie([vfieldl],pxv2701[1],0)
#%%
smltn2701_2=RunState(80,2000,64,64,5,1,0.25,5,20)
vectors2701_2=Resultant_Vectors(smltn2701_2,outv=True)
#%%
r=64
pxv2701_2=PixelatedVectors(smltn2701_2,vectors2701_2,r,r)
#%%
tf=TimeLoc(pxv2701_2[0],pxv2701_2[1],smltn2701_2)
print(tf)
#%%
#MovieNodes(smltn2701,14)
VectorMovie(pxv2701_2[0],pxv2701_2[1],9)
#%%
q_all=[]
for i in range(r**2):
    q=focal_quality_indicator2(pxv2701_2[0],pxv2701_2[1],pxv2701_2[1][i],49,1.5,5)
    q_all.append(q)

loci=q_all.index(max(q_all))
print(loci%r,loci//r)

