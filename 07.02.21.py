# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:39:19 2021

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
__import__=('09.02.21_code')
#%%
smltn0302=RunState(80,1000,32,32,5,1,0.35,3,20)
vectors0302=Resultant_Vectors(smltn0302,outv=True)
pxv0302=PixelatedVectors(smltn0302,vectors0302,32,32)
#%%
#MovieNodes(smltn0302,None)
writergif = animation.PillowWriter()
MovieNodes(smltn0302,None).save('testcase1102.gif',writer=writergif)
#%%
xdim=32
tM=TimeLoc(pxv0302[0],pxv0302[1],smltn0302)
print(tM)
#%%
locall=[0]*xdim**2
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn0302[0])-5):
        print(t)
        for i in range(32**2):
            q=focal_quality_indicator(pxv0302[0],pxv0302[1],pxv0302[1][i],t,2,5)
            locall[i]+=q
#%%visualising total dot product in tissue
matrix=np.zeros([32,32])
for i in range(len(locall)):
    y=i//xdim
    x=i%xdim
    matrix[y][x]=locall[i]
fig=plt.figure()
s=plt.imshow(matrix,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
fig.colorbar(s)
plt.savefig('1102alldot',dpi=1000)
#%%visualing top dot product values
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
matrix2=np.zeros([32,32])
for i in locall:
    matrix2[i[1]][i[0]]+=1
fig2=plt.figure()
s=plt.imshow(matrix2,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
fig2.colorbar(s)
#plt.savefig('1102maxdot',dpi=1000)
#%%need to test the premature firing method for cells
#initially we will compare the initial firing at the node level with our established 
#method which uses dot product
smltn0902=RunState(80,2000,64,64,5,1,0.3,5,20)
vectors0902=Resultant_Vectors(smltn0902,outv=True)
pxv0902=PixelatedVectors(smltn0902,vectors0902,64,64)
#%%
MovieNodes(smltn0902,15)
#%%
loc=miss_fire2(smltn0902,20,5,5)
print(smltn0902[1][loc[0]])
print(loc[1])
#%%comparing the dot product with premature firing method
tM=TimeLoc(pxv0902[0],pxv0902[1],smltn0902)
tau=5
vcond=4.75
#VectorMovie(pxvM27[0],pxvM27[1],tM)
if tM==0:
    locM=[0,0]
else:
    q_all=[]
    for i in range(64**2):
        q=focal_quality_indicator(pxv0902[0],pxv0902[1],pxv0902[1][i],tM,vcond,tau)
        q_all.append(q)
    loci=q_all.index(max(q_all))
    locM=[loci%64,loci//64]
#%%
it=50
spatdiff=[]
for i in range(it):
    print(i)
    smltn0902=RunState(80,2000,64,64,5,1,0.3,5,20)
    vectors0902=Resultant_Vectors(smltn0902,outv=True)
    pxv0902=PixelatedVectors(smltn0902,vectors0902,64,64)
    #premature method
    locdata=miss_fire2(smltn0902,20,5,5)
    spaloc=smltn0902[1][locdata[0]]
    #dot product method
    tM=TimeLoc(pxv0902[0],pxv0902[1],smltn0902)
    if tM==0:
        locM=[0,0]
    else:
        q_all=[]
        for i in range(64**2):
            q=focal_quality_indicator(pxv0902[0],pxv0902[1],pxv0902[1][i],tM,vcond,tau)
            q_all.append(q)
        loci=q_all.index(max(q_all))
        locM=[loci%64,loci//64]
        
    disti=np.sqrt((locM[0]-int(spaloc[0]))**2+(locM[1]-int(spaloc[1]))**2)
    spatdiff.append(disti)
ss=np.mean(spatdiff)/(4.75*5)
print(ss)
#%%
def miss_firecells(x,T,variation,xdim):
    data=Pixelation(x,xdim,xdim)[2]
    #time_difference
    miss_fire_list = [0]*len(x[0])
    data_fault=[]
    
    for cell in range(xdim**2):
        time_fired = -1
        delta_time = 0
        n=0
        for time in range(len(data[0])-1):
            if data[cell][time]>0 and n==0:
                tau=data[cell][time]
                n+=1
#            if time>0 and data[cell][time]>0 and data[cell][time+1]<data[cell][time] and data[cell][time-1]==0:                
            if time>0 and data[cell][time]>tau*0.9 and data[cell][time+1]<data[cell][time] and data[cell][time-1]<0.2*tau:                
                if time_fired > -1:         
                    delta_time = time - time_fired                    
                    if np.abs(delta_time - T) > variation:
                        miss_fire_list[time] = miss_fire_list[time] + 1
                        data_fault.append(time)
                        data_fault.append(cell)
                    time_fired = time
            
                if time_fired == -1:
                    time_fired = time
            if time==0 and data[cell][time]>0 and data[cell][time+1]<data[cell][time]:
                if time_fired > -1:         
                    delta_time = time - time_fired                    
                    if np.abs(delta_time - T) > variation:
                        miss_fire_list[time] = miss_fire_list[time] + 1
                        data_fault.append(time)
                        data_fault.append(cell)
                    time_fired = time
            
                if time_fired == -1:
                    time_fired = time                
        
    miss_fire_list_ratio = []
    for i in range(len(miss_fire_list)):
        miss_fire_list_ratio.append(miss_fire_list[i]/xdim**2)
    for i in range(len(miss_fire_list_ratio)):
        if miss_fire_list_ratio[i]>0 and miss_fire_list_ratio[i]>miss_fire_list_ratio[i-1] and miss_fire_list_ratio[i]<miss_fire_list_ratio[i+1] and miss_fire_list_ratio[i]<miss_fire_list_ratio[i+2]:
            t_ident=i
            break
        elif i==len(data[0])-1:
            t_ident=0
            
    time_fault=data_fault[::2]
    node_fault=data_fault[1::2]
    time_fault_index=[]
    for i in range(len(time_fault)):
        if time_fault[i]==t_ident:
            time_fault_index.append(i)
    node_fault_loc=[]
    for i in time_fault_index:
        node_fault_loc.append(node_fault[i])
    
    plt.plot(np.arange(len(data[0])),miss_fire_list_ratio,'x')
    plt.xlabel("Time")
    plt.ylabel("Miss Fire Ratio")
    plt.show()
    print(node_fault_loc[0])
    if t_ident!=0:
        loc=[node_fault_loc[0]%xdim,node_fault_loc[0]//xdim]
    else:
        loc=[0,0]
    return t_ident,loc
#%%smltncell
smltn0102=RunState(80,2000,64,64,5,1,0.3,5,20)
vectors0102=Resultant_Vectors(smltn0102,outv=True)
#%%
MovieNodes(smltn0102,6)
#%%
r=32#set the resolution u want
px0102=Pixelation(smltn0102,r,r)
pxv0102=PixelatedVectors(smltn0102,vectors0102,r,r)
#%%
VectorMovie(pxv0102[0],pxv0102[1],49)
#%%
t0102=np.arange(0,80,1)
plt.plot(t0102,px0102[2][840])#plot of the time series of a cell in grid
plt.xlabel('Time')
plt.ylabel('Cell activation')
#plt.savefig('testpremcell',dpi=1000)
#%%
sa=miss_firecells(smltn0102,20,5,32)
print(sa)
ss=miss_fire2(smltn0102,20,5,5)
print(smltn0102[1][ss[0]])
#%%
res=np.arange(6,66,2)
it=100
celldis=[[] for i in res]
for i in range(it):
    print(i)
    smltn1002=RunState(80,2000,64,64,5,1,0.3,5,20)
    nodenumber=miss_fire2(smltn1002,20,5,5)[0]
    if nodenumber==None:
        continue
    else:
        node=np.array(smltn1002[1][nodenumber])
    for r in range(len(res)):
        b=64/res[r]
        cell=miss_firecells(smltn1002,20,5,res[r])[1]
        cellacc=node//b
        
        dis=np.sqrt((cell[0]-cellacc[0])**2+(cell[1]-cellacc[1])**2)
        celldis[r].append(dis)
#%%
#np.savetxt('1002premmethod.txt',celldis)
#%%preparing premature data
celldis=np.loadtxt('1002premmethod.txt')
res=np.arange(6,66,2)
it=len(celldis[0])
vcond=4.75
celldismean=np.array([np.mean(i) for i in celldis])
celldisstd=np.array([np.std(i) for i in celldis])/np.sqrt(it)
vcondal=np.array([vcond*i/64 for i in res])
pathl=vcondal*5
celldisnorm=celldismean/pathl
celldisnormsd=celldisstd/pathl
baxis=64/res#bcoefficient
#%%preparing data from dot product method
res=np.arange(6,64,2)
vcond=4.75
d_allold=np.loadtxt('spacediffrand3101old.txt')
d_allnew=np.loadtxt('spacediffrand3101new.txt')
d_all=np.concatenate([d_allold,d_allnew])[:-12]
d_allr=np.loadtxt('spacediffrand0102new.txt')[:-12]
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
plt.plot(res,celldisnorm[:-1],'x',label='Premature firing')
plt.errorbar(res,celldisnorm[:-1],yerr=celldisnormsd[:-1],fmt='.',capsize=5)
plt.hlines(1,1,64)
plt.plot(res,dtow,'o',label='Dot Product')
plt.errorbar(res,dtow,yerr=dtowsd,fmt='.',capsize=5)
plt.plot(res,dtowr,'o',label='Random Vectors')
plt.errorbar(res,dtowr,yerr=dtowsdr,fmt='.',capsize=5)
plt.xlabel('Resolution')
plt.ylabel('Cell distance to path length ratio')
plt.title('Testing the capabilities of Premature firing method')
plt.legend()
#plt.savefig('1002',dpi=1000)

