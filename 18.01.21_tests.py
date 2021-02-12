# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:34:19 2021

in the following script i aim to measure the conduction velocity
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
import copy
from scipy.optimize import curve_fit
__import__=('18.01.21_code')
#%%model to be used
data=RunState(100,2000,64,64,5,1,0.1,5,100,1)
MovieNodes(data,None)
#%%this function calculates the time needed for wave to propagate through the whole plane
#it will be used to calculate conduction velocity
for t in range(len(data[0])):
    if set(data[0][t])=={0}:
        print(t+1)#accounts for the fact that 
        break
#%%need to auromate the above for radius=5
time=[]
cv=[]
for i in range(100):
    data=RunState(100,2000,64,64,5,1,0.1,5,100,1)
    for t in range(len(data[0])):
        if set(data[0][t])=={0}:
            time.append(t+1)#accounts for the fact that 
            cv.append(64/(t+1))
            break

time_m=np.mean(time)
time_s=np.std(time)
cv_m=np.mean(cv)
cv_s=np.std(cv)
print(time_m, '+-',time_s)
print(cv_m, '+-',cv_s)
#%%testing the limits over a range of radii
data=RunState(100,2000,64,64,5,1,0.1,12,100,1)
MovieNodes(data,None)
#%%collecting data for conduction velocity versus radius plot
radii=np.arange(2,8.2,0.2)
velocm=[]
velocs=[]
for r in radii:
    print(r)
    veloc_i=[]
    for i in range(100):
        data=RunState(100,2000,64,64,5,1,0.2,r,100,1)
        for t in range(len(data[0])):
            if set(data[0][t])=={0}:
                veloc_i.append(64/(t+1))#accounts for the fact that 
                break
            elif set(data[0][t])!={0} and t==99:#condition for cases where the wave does
                #not propagate across area 
                veloc_i.append(0)
    velocm.append(np.mean(veloc_i))
    velocs.append(np.std(veloc_i))
#%%plotting the above
vel_m2=np.array(velocm)    
vel_e2=np.array(velocs)/np.sqrt(len(velocs))    
#np.savetxt('cvdata0.2.txt',vel_m)
#np.savetxt('cvdataerr0.2.txt',vel_e)
#%%
radii=np.arange(2,8.2,0.2)
vel_m1=np.loadtxt('cvdata0.1.txt')
vel_e1=np.loadtxt('cvdataerr0.1.txt')
vel_m2=np.loadtxt('cvdata0.2.txt')
vel_e2=np.loadtxt('cvdataerr0.2.txt')
#%%checking theoretical value of vc vs radius
def thrvcr(r,epsilon):
    n=r*(1-epsilon)
    d=1+epsilon
    return n/d
radiit1=np.arange(2,8.2,0.1)
vct1=thrvcr(radiit1,0.9)
#%%
plt.plot(radii,vel_m1,'x',label=r'$\epsilon=0.1$')
plt.errorbar(radii,vel_m1,yerr=vel_e1,fmt='.',capsize=5)
plt.plot(radii,vel_m2,'o',label=r'$\epsilon=0.2$')
plt.errorbar(radii,vel_m2,yerr=vel_e2,fmt='.',capsize=5)
#plt.plot(radiit1,vct1,label='Theoretical')
plt.xlabel('Radius, R')
plt.ylabel('Conduction velocity, v')
plt.legend()
#plt.savefig('cvvsradius',dpi=1000)
#%%testing james divergent code
#we need to prepare slicing method fit parameters
std_tot=np.loadtxt('std_13.01.21.txt')
erv=len(std_tot[0])
std_gr=[np.mean(i) for i in std_tot]
std_grer=[np.std(i)/erv for i in std_tot]
epsilon=np.arange(0,0.6,0.01)
#%%fitting the data for the sinus rhythm region
#based on initial inspection we choose on how many points to include
#in linear fit
fit_std,cov_std=np.polyfit(epsilon[:17],std_gr[:17],1,w=1/np.array(std_grer[:17]),cov=True)
pfit=np.poly1d(fit_std)#fit for sinus rhythm for average vector
errorgrad=np.sqrt(cov_std[0,0])#error in gradient
errorc=np.sqrt(cov_std[1,1])#error in intercept
errors_avg=np.array([errorgrad,errorc])
#find worst case scenario to distinguish between SR and AF
fit_stdw=fit_std+20*errors_avg#worst case fit parameters
pfitw=np.poly1d(fit_stdw)
#%%visualising slicing distribution
plt.plot(epsilon,std_gr,'o',label='Data')
plt.errorbar(epsilon,std_gr,yerr=std_grer,fmt='.',capsize=5)
plt.plot(epsilon,pfit(epsilon),label='Linear fit')
plt.plot(epsilon,pfitw(epsilon),label='Max accepted fit')
plt.xlabel(r'Epsilon,$\epsilon$')
plt.ylabel('Standard deviation of divergence')
plt.legend()
#%%
data=RunState(80,2000,64,64,5,1,0.22,5,20,1)
vec_b=Resultant_Vectors(data,inv=True,outv=True)
pxv_64=PixelatedVectors(data,vec_b,64,64)
pxv_62=PixelatedVectors(data,vec_b,62,62)
pxv_48=PixelatedVectors(data,vec_b,48,48)
pxv_32=PixelatedVectors(data,vec_b,32,32)
#%%
MovieNodes(data,None)
#writergif = animation.PillowWriter()
#MovieNodes(data,None).save('perfect_source_vector_animation1.gif',writer=writergif)
#%%
pxv_64=PixelatedVectors(data,vec_b,64,64)
loc=AFIdentifyBoth(pxv_64[0],pxv_64[1],data,fit_std,errors_avg,64,64)[0]
#print(loc)
#iloc=loc[0][1]*64+loc[0][0]
#quality = focal_quality_indicator(pxv_64[0],pxv_64[1],64,64,1,1,iloc,,2.9,3)
#%%
xdim=64
mdiv=int(xdim**2*0.01)
dottop=[]
alloc=[]
for i in range(26,27):
    iloc=loc[i][0][1]*xdim+loc[i][0][0]
    itime=loc[i][1]
    quality = focal_quality_indicator(pxv_64[0],pxv_64[1],64,64,1,1,iloc,itime,3,5)
    dottop.append(quality)
    alloc.append(iloc)
ind=dottop.index(max(dottop))
locbest=alloc[ind]
print(locbest%xdim,locbest//xdim) 
print(itime)  
#print(dottop,alloc)
VectorMovie(pxv_64[0],pxv_64[1],itime)
plt.savefig('locationm',dpi=1000)
#VectorMovie(pxv_64[0],pxv_64[1],itime+2)
#%%
loc=AFIdentifyBoth(pxv_62[0],pxv_62[1],data,fit_std,errors_avg,64,62)[0]
#print(loc)
#%%
xdim=62
mdiv=int(xdim**2*0.01)
dottop=[]
alloc=[]
for i in range(mdiv):
    iloc=loc[i][0][1]*xdim+loc[i][0][0]
    itime=loc[i][1]
    quality = focal_quality_indicator(pxv_62[0],pxv_62[1],62,62,64/62,64/62,iloc,itime,2.9,5)
    dottop.append(quality)
    alloc.append(iloc)
ind=dottop.index(max(dottop))
locbest=alloc[ind]
print(locbest%xdim,locbest//xdim) 
print(itime)
VectorMovie(pxv_62[0],pxv_62[1],itime)  
#print(dottop,alloc)
#%%
pxv_60=PixelatedVectors(data,vec_b,60,60)
loc=AFIdentifyBoth(pxv_60[0],pxv_60[1],data,fit_std,errors_avg,64,60)[0]
#print(loc)
#%%
xdim=60
mdiv=int(xdim**2*0.01)
dottop=[]
alloc=[]
for i in range(mdiv):
    iloc=loc[i][0][1]*xdim+loc[i][0][0]
    itime=loc[i][1]
    quality = focal_quality_indicator(pxv_60[0],pxv_60[1],60,60,64/60,64/60,iloc,itime,2.9,5)
    dottop.append(quality)
    alloc.append(iloc)
ind=dottop.index(max(dottop))
locbest=alloc[ind]
print(locbest%xdim,locbest//xdim) 
print(itime)
VectorMovie(pxv_60[0],pxv_60[1],itime)  
plt.savefig('locatiocg',dpi=1000)
#print(dottop,alloc)
#%%
loc=AFIdentifyBoth(pxv_48[0],pxv_48[1],data,fit_std,errors_avg,64,48)[0]
len(loc)
#%%
xdim=48
mdiv=int(xdim**2*0.01)
dottop=[]
alloc=[]
for i in range(19,20):
    iloc=loc[i][0][1]*xdim+loc[i][0][0]
    itime=loc[i][1]
    quality = focal_quality_indicator(pxv_48[0],pxv_48[1],48,48,64/48,64/48,iloc,itime,2.25,5)
    dottop.append(quality)
    alloc.append(iloc)
ind=dottop.index(max(dottop))
locbest=alloc[ind]
print(locbest%xdim,locbest//xdim) 
print(itime)  
print(dottop,alloc)
VectorMovie(pxv_48[0],pxv_48[1],itime)
#VectorMovie(pxv_48[0],pxv_48[1],itime+5)
#%%
loc=AFIdentifyBoth(pxv_32[0],pxv_32[1],data,fit_std,errors_avg,64,32)[0]
#print(loc)
#%%
xdim=32
mdiv=int(xdim**2*0.01)
dottop=[]
alloc=[]
for i in range(mdiv):
    iloc=loc[i][0][1]*xdim+loc[i][0][0]
    itime=loc[i][1]
    quality = focal_quality_indicator(pxv_32[0],pxv_32[1],32,32,1,1,iloc,itime,1.5,5)
    dottop.append(quality)
    alloc.append(iloc)
ind=dottop.index(max(dottop))
locbest=alloc[ind]
print(locbest%xdim,locbest//xdim) 
print(itime)  
#print(dottop,alloc)
VectorMovie(pxv_32[0],pxv_32[1],itime)
#%%
res=np.arange(30,66,2)
d_r=[]
for i in range(50):
    print(i)
    data_r=RunState(80,2000,64,64,5,1,0.27,5,20,1)
    vec_r=Resultant_Vectors(data_r,inv=True,outv=True)
    pxv_64=PixelatedVectors(data_r,vec_r,64,64)
    
    loc=AFIdentifyBoth(pxv_64[0],pxv_64[1],data_r,fit_std,errors_avg,64,64)[0]    
    xdim=64 
    vc=2.9
    mdiv=int(xdim**2*0.01)
    dottop=[]
    alloc=[]
    for i in range(mdiv):
        iloc=loc[i][0][1]*xdim+loc[i][0][0]
        itime=loc[i][1]
        quality = focal_quality_indicator(pxv_64[0],pxv_64[1],64,64,1,1,iloc,itime,vc,5)
        dottop.append(quality)
        alloc.append(iloc)
    ind=dottop.index(max(dottop))
    locbest=alloc[ind]
    locs_mf=np.array([locbest%xdim,locbest//xdim])
    
    di_r=[]
    for r in range(len(res)):
        print(res[r])
        xdim_cg=res[r] 
        b=xdim/xdim_cg
        pxv_r=PixelatedVectors(data_r,vec_r,res[r],res[r])
        
        loc_cg=AFIdentifyBoth(pxv_r[0],pxv_r[1],data_r,fit_std,errors_avg,64,res[r])[0]            
        mdiv_cg=int(xdim_cg**2*0.01)
        dottop=[]
        alloc=[]
        for i in range(mdiv_cg):
            iloc_cg=loc_cg[i][0][1]*xdim_cg+loc_cg[i][0][0]
            itime_cg=loc_cg[i][1]
            quality = focal_quality_indicator(pxv_r[0],pxv_r[1],res[r],res[r],b,b,iloc_cg,itime_cg,vc/b,5)
            dottop.append(quality)
            alloc.append(iloc_cg)
        ind=dottop.index(max(dottop))
        locbest=alloc[ind]
        locs_cgf=[locbest%xdim_cg,locbest//xdim_cg]
        
        locs_mcg=locs_mf//b
        #print(locs_mcg,locs_mf,locs_cgf)
        dist_i=np.sqrt((locs_mcg[0]-locs_cgf[0])**2+(locs_mcg[1]-locs_cgf[1])**2)
        #dist_f=dist_i*b
        di_r.append(dist_i)
    d_r.append(di_r) 
#%%
#np.savetxt('disttest36reps.txt',d_r)
#np.savetxt('disttest36reps2.txt',d_r)
#%%
d_r1=np.loadtxt('disttest36reps.txt')
d_r2=np.loadtxt('disttest36reps2.txt')
d_r=np.concatenate((d_r1,d_r2))
#%%
d_rn=[]
for j in range(len(d_r[0])):
    d_rni=[]
    for i in range(len(d_r)):
        d_rni.append(d_r[i][j])
    d_rn.append(d_rni)
d_mean=np.array([np.mean(i) for i in d_rn])
d_sd=np.array([np.std(i) for i in d_rn])/np.sqrt(86)
vcall=np.array([vc*i/64 for i in res])
pathl=5*vcall
d_norm=d_mean/pathl
d_normsd=d_sd/pathl
xaxis=np.array([64]*len(res))/res
#%%
plt.hlines(1,1,2.3)
plt.plot(xaxis,d_norm,'x')
plt.errorbar(xaxis,d_norm,yerr=d_normsd,fmt='.',capsize=5)
plt.xlabel('b factor')
plt.ylabel('Normalised distance between points identified')        
plt.savefig('resdistribution',dpi=1000)