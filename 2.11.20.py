# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:45:52 2020

@author: ahadj
apply changes discussed in last meeting
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

def round_down(n, decimals=0):#used to identify in which cell a node belongs to
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def GenerateCoorArray(N,x_size,y_size):#prepares a list with the coordinates of all nodes
    pmcell=int(N/10)#number of pacemaker cells
    CoorStore = []
    for i in range(0,pmcell):#sets the first 10% of nodes as pacemakers
        Coor = []
        Coor.append(0)
        Coor.append(random.random()*y_size)
        CoorStore.append(Coor) 
    for i in range(pmcell,N):   
        Coor = []
        Coor.append(random.random()*x_size)
        Coor.append(random.random()*y_size)
        CoorStore.append(Coor) 
    return sorted(CoorStore , key=lambda k: [k[0], k[1]])

def distance_checker(R,i,j,CoorStore,y_size):#checks the distance between 2 nodes 
    #in a way to allow periodic BCs in y-direction and open in x-direction
    diff=CoorStore[i][1]-CoorStore[j][1]
    if abs(diff)<=R:# for rest cases
        #print('here')
        d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    elif abs(diff)>R:    
        if CoorStore[i][1]+R>y_size:#sets upper BC
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]-y_size - CoorStore[j][1]) **2 )**0.5
            #print('here1')
        elif CoorStore[i][1]-R<0:#sets lower BC
            #print('here2')
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]+y_size - CoorStore[j][1]) **2 )**0.5
        else:
            #print('here3')
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    if d <= R:
        return True
    else:
        return False       
 
       
def ConnectionArray(N,x_size,y_size,CoorStore,R):
    #a list containing all the connections of the ith node in list
    ConnectionsStore = []
    for i in range(N):
        connections = []
        nodecoor=CoorStore[i]
        ll=nodecoor[0]-R
        rl=nodecoor[0]+R
        for j in range(N):
            if i !=j and ll<=CoorStore[j][0]<=rl:    
                #print(i,j)
                if distance_checker(R,i,j,CoorStore,y_size) == True:
                    connections.append(j)
        ConnectionsStore.append(connections)
    return ConnectionsStore

def FunctionalArray(N,x_size,y_size,tau,delta,epsilon):
    #prepares two lists: one with refractory period and one with excitation
    #probability of each node
    taustore = []
    epsilonstore = []
    for i in range(N):
        taustore.append(tau)
        if random.random() < delta:
            epsilonstore.append(epsilon)
        else:
            epsilonstore.append(0)
    functionalstore  =[]
    functionalstore.append(taustore)
    functionalstore.append(epsilonstore)
    return functionalstore

def GenerateState(N):#prepares list with state of each node
    Statelist=[]
    for i in range(N):
        Statelist.append(0)
    return Statelist
#%%      
def PaceMaker(StateListP,FunctionListP,N):#excites pacemaker cells every T time steps
    pmcell=int(N/10)
    for i in range(pmcell):
        rand=random.random()
        if rand>FunctionListP[1][i]:
            StateListP[i]= FunctionListP[0][i] + 1                
    return StateListP

def UpdateStateLattice(N, StateListU, FunctionListU, Connection):
#updates the state of each node  
    Updatedlist = []
    for i in range(N):
        Updatedlist.append(0)
    for i in range(N):
        if StateListU[i] > 0: #reduce refractory period countdown by one
            Updatedlist[i] = StateListU[i]- 1
            continue
            
        if StateListU[i]== 0: #here check if any neighbours are in firing state
            rand=random.random()
            for j in range(len(Connection[i])):
                NeighNode=Connection[i][j]
                if StateListU[NeighNode] == FunctionListU[0][NeighNode]+1:
                    if rand>FunctionListU[1][NeighNode]:
                        Updatedlist[i]=FunctionListU[0][i]+1
    return Updatedlist

def RunState(TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T):
    #runs the model for a specified number of timesteps
    #and returns a list with all timesteps
    SetupS=timeit.default_timer()
    Coordinates = GenerateCoorArray(N,x_size,y_size)
    Connections = ConnectionArray(N,x_size,y_size,Coordinates,r)
    FunctionListR = FunctionalArray(N,x_size,y_size,tau,delta,epsilon)
    StateListR = GenerateState(N)
    SetupE=timeit.default_timer()
    
    StateStore = []
    RunS=timeit.default_timer()
    for i in range(TimeSteps):        
        if i == 0 or i % T == 0:
            StateListR = PaceMaker(StateListR,FunctionListR,N)
            StateStore.append(StateListR)
        else:
            StateListR = UpdateStateLattice(N,StateListR,FunctionListR,Connections )
            StateStore.append(StateListR)
    RunE=timeit.default_timer()
    
    TimeS=SetupE-SetupS
    TimeR=RunE-RunS
    data=[TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T]
    return StateStore, Coordinates, Connections,TimeS,TimeR,data

def MovieNodes(a):
    #prepares movie for evolution of state of nodes
    x=[]
    y=[]
    for i in range(len(a[1])):
        x.append(a[1][i][0])
        y.append(a[1][i][1])
    
    fig=plt.figure()
    
    ims=[]
    for i in range(len(a[0])):
        im=plt.scatter(x,y,c=a[0][i],edgecolors='r',cmap=plt.cm.binary)
        ims.append([im])
    plt.colorbar()
    
    ani = animation.ArtistAnimation(fig, ims, interval=100, 
                                    repeat_delay=1000)
    return ani
    
def Pixelation(cc,x_grid_size,y_grid_size):
    #prepares pixelated movie based on resolution requested
    x_size=cc[5][2]
    y_size=cc[5][3]
    tau=cc[5][4]
    
    grid_coor = []
    for i in range(int(x_grid_size)):
        for j in range(int(y_grid_size)):
            grid_coor.append([i,j])
    
    grid_container = []
    timeseries=[]#contains time-series for each cell
    for i in range(len(grid_coor)):
        grid_container.append([])
        timeseries.append([])
    for i in range(len(cc[1])):
        grid_coor_state = round_down(cc[1][i][0]/(x_size/x_grid_size)), round_down(cc[1][i][1]/(y_size/y_grid_size)) 
        grid_container[int(grid_coor_state[0]*(x_grid_size) + grid_coor_state[1] )].append(i)
    
    allgridvalues=[]
    for i in range(len(cc[0])):
        grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
        for cell in range(len(grid_container)):
            sum = 0
            for node in range(len(grid_container[cell])):
                sum = sum + cc[0][i][grid_container[cell][node]]
            grid_sum[y_grid_size-1-grid_coor[cell][1]][grid_coor[cell][0]] = sum
            timeseries[cell].append(sum)
        allgridvalues.append(grid_sum)                
    
    nodespc=[]#nodespercell(determining cell with max number of nodes)
    for i in range(len(grid_container)):
        nodespc.append(len(grid_container[i]))
    maxcellcolor=np.mean(nodespc)*(tau+1)#determining max value possible 
    #in grid_sum,required to set the color scale
    return allgridvalues,int(maxcellcolor) ,timeseries

def MoviePixels(pixeldata):
    Allgridvalues=pixeldata[0]
    fig = plt.figure()
    ims=[]    
    for i in range(len(Allgridvalues)):
        im=plt.imshow(Allgridvalues[i],interpolation='none',cmap=plt.cm.binary,vmin=0,vmax=pixeldata[1],animated=True)
        if i==0:
            fig.colorbar(im)
        ims.append([im])
    
    plt.title('Pixelated Grid')
    ani = animation.ArtistAnimation(fig, ims, interval=100, 
                                    repeat_delay=1000)
    return ani

#%%prepare data
a=RunState(100,8000,128,128,5,1,0.22,4,20)
MovieNodes(a)
#%%pixelating the data
sss=Pixelation(a,8,8)
MoviePixels(sss)
#%%prepare data in panda form
dfi=pd.DataFrame(sss[2])
dfe=dfi.T
print(dfe)
#%%
maxlag=7
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    #df.columns = [var + '_x' for var in variables]
    #df.index = [var + '_y' for var in variables]
    return df
#%%performing the Granger Causality test
grangers_causation_matrix(dfe, variables = dfe.columns)  
#%%trying out james method
def BoundaryChecker(i,j, y_size):
    if i == y_size - 1:
        return -1 
    else:
        return i

def ExcitationVector(PixelDataAll):
    PixelData=PixelDataAll[0]
    vectors_store=[]
    
    for t in range(2,len(PixelData)-1):
        x_size=len(PixelData[0][0])
        y_size=len(PixelData[0])
        pixelcoor=[]
        for i in range(x_size):
            for j in range(y_size):
                pixelcoor.append([0,0])        
        
        for i in range(x_size):
            for j in range(y_size):
                j=y_size-1-j#to keep cartesian coordinate convention
                #first check if at previous time step t was excited
                if PixelData[t-2][j][i]<PixelData[t-1][j][i]<PixelData[t][j][i]:
                    
                    if i-1 != -1:# left boundary conditions
                        if PixelData[t-1][j][i-1]<PixelData[t][j][i-1]<PixelData[t+1][j][i-1]:
                            pixelcoor[i*x_size+j][0]-=1
                            
                    if i+1 != x_size:# left boundary conditions
                        if PixelData[t-1][j][i+1]<PixelData[t][j][i+1]<PixelData[t+1][j][i+1]:
                            pixelcoor[i*x_size+j][0]+=1
                       #below     
                    if PixelData[t-1][j-1][i]<PixelData[t][j-1][i]<PixelData[t+1][j-1][i]:
                        pixelcoor[i*x_size+j][1]-=1
                    
                    j = BoundaryChecker(j,i,y_size)#up boundary conditions
                    if PixelData[t-1][j+1][i]<PixelData[t][j+1][i]<PixelData[t+1][j+1][i]:
                        pixelcoor[i*x_size+j][1]+=1
        vectors_store.append(pixelcoor)
    return vectors_store
#%%
test= ExcitationVector(sss)                          
x_size=7 
y_size=7                          

qs=[]
fig=plt.figure()
for j in range(len(test)):
    x=[]
    y=[]
    u=[]
    v=[]
    for i in range(len(test[j])):
        u.append(test[j][i][0])
        v.append(test[j][i][1])
        x.append(i//x_size)
        y.append(i%y_size)
    
    q = plt.quiver(x,y,u,v,pivot='mid', angles='xy', scale_units='xy',scale=1)                   
    qs.append(q)   

ani = animation.ArtistAnimation(fig, qs, interval=100, 
                                    repeat_delay=1000)  
                    
        
