# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 00:17:27 2021

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
#%%

x = RunState(100,2000,64,64,5,1,0.28,5,20)
#%%
ani = MovieNodes(x,12)
#%%james code for locating the time at which AF initiates using the time series of nodes
#def miss_fire(x,T,Tau,variation):
#    data = x[0]
#    N = len(data[0])
#    print(N)
#    Tau_plus = Tau + 1
#    #time_difference
#    miss_fire_list = [0]*len(x[0])
#    hist = []    
#    
#    for node in range(N):
#        time_fired = -1
#        delta_time = 0
#        for time in range(len(data)):
#            if data[time][node] == Tau_plus:
#                
#                if time_fired > -1:
#         
#                    delta_time = time - time_fired
#                    
#                    if np.abs(delta_time - T) > variation:
#                        hist.append(delta_time)
#                        if node == 499:
#                            print(delta_time,variation,node)
#                        miss_fire_list[time] = miss_fire_list[time] + 1
#                            
#                    time_fired = time
#            
#                if time_fired == -1:
#                    time_fired = time
#        
#    miss_fire_list_ratio = []
#    for i in range(len(miss_fire_list)):
#        miss_fire_list_ratio.append(miss_fire_list[i]/N)
#        
#    #print(len(miss_fire_list_ratio),len(np.arange(data)))
#    time = np.arange(len(data))
#        
#    data_ = {'ratio':miss_fire_list_ratio,'time':time}
#    df = pd.DataFrame(data_)
#    df['rolling'] = df['ratio'].rolling(window=10).mean()
#    
#    plt.plot(df['time'],df['rolling'] )
#    plt.figure()
#    
#    plt.plot(np.arange(len(data)),miss_fire_list_ratio)
#    plt.xlabel("time")
#    plt.ylabel("Miss Fire Ratio")
#    plt.show()
#    #plt.figure()
#    #plt.hist(hist)
#    return data_,df
#k,m=miss_fire(x,20,5,5)
#%%
smltn0102=RunState(80,2000,64,64,5,1,0.29,5,20)
vectors0102=Resultant_Vectors(smltn0102,outv=True)
#%%
MovieNodes(smltn0102,None)
#%%
writergif = animation.PillowWriter()
MovieNodes(smltn0102,None).save('testcase0102_2.gif',writer=writergif)
#%%
r=64#set the resolution u want
px0102=Pixelation(smltn0102,r,r)
pxv0102=PixelatedVectors(smltn0102,vectors0102,r,r)
#%%
t0102=np.arange(0,80,1)
#plt.plot(t0102,px0102[2][1162])#plot of the time series of a cell in grid
#%%
tsn0102=[]#contains the time series of each node
for c in range(2000):
    tsn0102.append([smltn0102[0][t][c] for t in t0102])
plt.plot(t0102,tsn0102[1162])#plot of time series of node
plt.xlabel('Time')
plt.ylabel('Node Activation')
plt.title('[33.7,21.0]')
#plt.savefig('locpoint',dpi=1000)
#%%testing miss_fire function so that it finds location as well
def miss_fire2(x,T,Tau,variation):
    data = x[0]
    N = len(data[0])
    Tau_plus = Tau + 1
    #time_difference
    miss_fire_list = [0]*len(x[0])
    hist = []    
    data_fault=[]
    
    for node in range(N):
        time_fired = -1
        delta_time = 0
        for time in range(len(data)):
            if data[time][node] == Tau_plus:                
                if time_fired > -1:         
                    delta_time = time - time_fired                    
                    if np.abs(delta_time - T) > variation:
                        hist.append(delta_time)
                        miss_fire_list[time] = miss_fire_list[time] + 1
                        data_fault.append(time)
                        data_fault.append(node)
                    time_fired = time
            
                if time_fired == -1:
                    time_fired = time
        
    miss_fire_list_ratio = []
    for i in range(len(miss_fire_list)):
        miss_fire_list_ratio.append(miss_fire_list[i]/N)
    for i in range(len(miss_fire_list_ratio)):
        if miss_fire_list_ratio[i]>0 and miss_fire_list_ratio[i]>miss_fire_list_ratio[i-1] and miss_fire_list_ratio[i]<miss_fire_list_ratio[i+1]:
            t_ident=i
            break
    time_fault=data_fault[::2]
    node_fault=data_fault[1::2]
    time_fault_index=[]
    for i in range(len(time_fault)):
        if time_fault[i]==t_ident:
            time_fault_index.append(i)
    node_fault_loc=[]
    for i in time_fault_index:
        node_fault_loc.append(node_fault[i])
    node_fault_loc_x=[x[1][i][0] for i in node_fault_loc]
    node_fault_loc_y=[x[1][i][1] for i in node_fault_loc]
    node_final=[np.mean(node_fault_loc_x),np.mean(node_fault_loc_y)]
    #print(len(data_fault),len(time_fault),len(node_fault))
    #print(len(miss_fire_list_ratio),len(np.arange(data)))
    time = np.arange(len(data))
        
    data_ = {'ratio':miss_fire_list_ratio,'time':time}
    df = pd.DataFrame(data_)
    df['rolling'] = df['ratio'].rolling(window=10).mean()
    
    
    plt.plot(df['time'],df['rolling'] )
    plt.xlabel("Time")
    plt.ylabel("Rolling average of miss fire ratio")    
    #plt.savefig('Missfireavg',dpi=1000)
    plt.figure()
    
    plt.plot(np.arange(len(data)),miss_fire_list_ratio)
    plt.xlabel("Time")
    plt.ylabel("Miss Fire Ratio")
    plt.show()
    #plt.savefig('MissFire',dpi=1000)
    #plt.figure()
    #plt.hist(hist)
    return t_ident,node_fault_loc,node_final
#%%testing miss_fire2 function
ss=miss_fire2(smltn0102,20,5,5)
print(ss[0],ss[1])
print(smltn0102[1][ss[1][0]])
#%%
MovieNodes(smltn0102,16)
plt.savefig('locattestlat',dpi=1000)
#%%testing small field
smltn0102_2=RunState(5,20,4,4,2,3,0.3,1,20)
vectors0102_2=Resultant_Vectors(smltn0102_2,inv=True)
#%%
MovieNodes(smltn0102_2,None)
#MovieNodes(smltn0102_2,4)
#%%
VectorMovieNodes(vectors0102_2,smltn0102_2[1],1)
#%%calculating the average conduction velocity
xdim=64
it=100
radii=np.arange(1,9)
vc_allmean1=[]
vc_allstd1=[]
for ri in radii:
    print(ri)
    vc_all=[]
    for i in range(it):
        smltn0102r=RunState(35,2000,64,64,5,1,0.3,ri,100)
        minloc=min([i for i in range(len(smltn0102r[1])) if smltn0102r[1][i][0]>xdim-1])#finds 
        for t in range(len(smltn0102r[0])):
            if set(smltn0102r[0][t][minloc:])!={0}:
                vc=xdim/t
                vc_all.append(vc)
                #print(t)
                break
    if len(vc_all)==0:
        vc_allmean1.append(0)
        vc_allstd1.append(0)
    else:
        vc_allmean1.append(np.mean(vc_all))
        vc_allstd1.append(np.std(vc_all)/np.sqrt(it))
#print(vc_allmean,vc_allstd)
vc_allmean=[]
vc_allstd=[]
for ri in radii:
    print(ri)
    vc_all=[]
    for i in range(it):
        smltn0102r=RunState(35,2000,64,64,5,1,0.1,ri,100)
        minloc=min([i for i in range(len(smltn0102r[1])) if smltn0102r[1][i][0]>xdim-1])#finds 
        for t in range(len(smltn0102r[0])):
            if set(smltn0102r[0][t][minloc:])!={0}:
                vc=xdim/t
                vc_all.append(vc)
                #print(t)
                break
    if len(vc_all)==0:
        vc_allmean.append(0)
        vc_allstd.append(0)
    else:
        vc_allmean.append(np.mean(vc_all))
        vc_allstd.append(np.std(vc_all)/np.sqrt(it))
#%%saving data
np.savetxt('condvelvsradius0.3',vc_allmean)
np.savetxt('condvelvsradius0.3',vc_allstd)
#%%
vc_allstd[7]=10**-18
#%%fitting the data
fit_vc,cov_vc=np.polyfit(radii[2:-2],vc_allmean[2:-2],1,cov=True)
pfit=np.poly1d(fit_vc)#fit for sinus rhythm for average vector
errorgrad=np.sqrt(cov_vc[0,0])#error in gradient
errorc=np.sqrt(cov_vc[1,1])#error in intercept
print(fit_vc[0],r'$\pm$',errorgrad)
#%%
plt.plot(radii,vc_allmean,'x')
plt.errorbar(radii,vc_allmean,yerr=vc_allstd,fmt='.',capsize=5)
plt.plot(radii,pfit(radii))
#%%
