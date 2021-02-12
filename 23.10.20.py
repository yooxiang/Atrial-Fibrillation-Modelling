# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:42:47 2020

@author: ahadj

checking james code on resolution animation 
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit

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
    return CoorStore

def distance_checker(R,i,j,CoorStore,y_size):#checks the distance between 2 nodes 
    #in a way to allow periodic BCs in y-direction and open in x-direction
    if CoorStore[i][1]+R>100:#sets upper BC
        d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]-y_size - CoorStore[j][1]) **2 )**0.5
        if d < R:
            return True
        else:
            return False
    elif CoorStore[i][1]-R<0:#sets lower BC
        d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]+y_size - CoorStore[j][1]) **2 )**0.5
        if d < R:
            return True
        else:
            return False       
    else:# for rest cases
        d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
        if d < R:
            return True
        else:
            return False 
       
def ConnectionArray(N,x_size,y_size,CoorStore,R):
    #a list containing all the connections of the ith node in list
    ConnectionsStore = []
    for i in range(N):
        connections = []
        for j in range(N):
            if i !=j:
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
    SetupS=timeit.timeit()
    Coordinates = GenerateCoorArray(N,x_size,y_size)
    Connections = ConnectionArray(N,x_size,y_size,Coordinates,r)
    FunctionListR = FunctionalArray(N,x_size,y_size,tau,delta,epsilon)
    StateListR = GenerateState(N)
    SetupE=timeit.timeit()
    
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
    return StateStore, Coordinates, Connections,TimeS,TimeR

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
    
def MoviePixels(cc,x_size,y_size,x_grid_size,y_grid_size):
    #prepares pixelated movie based on resolution requested
    grid_coor = []
    for i in range(int(x_grid_size)):
        for j in range(int(y_grid_size)):
            grid_coor.append([i,j])
    
    grid_container = []
    for i in range(len(grid_coor)):
        grid_container.append([])
    for i in range(len(cc[1])):
        grid_coor_state = round_down(cc[1][i][0]/(x_size/x_grid_size)), round_down(cc[1][i][1]/(y_size/y_grid_size)) 
        grid_container[int(grid_coor_state[0]*(x_grid_size) + grid_coor_state[1] )].append(i)
    
    fig = plt.figure()
    ims=[]
    for i in range(len(cc[0])):
        grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
        for cell in range(len(grid_container)):
            sum = 0
            for node in range(len(grid_container[cell])):
                sum = sum + cc[0][i][grid_container[cell][node]]
            grid_sum[grid_coor[cell][1]][grid_coor[cell][0]] = sum
        
        
        im=plt.imshow(grid_sum,interpolation='none',cmap=plt.cm.binary,animated=True)
        ims.append([im])
        
    ani = animation.ArtistAnimation(fig, ims, interval=100, 
                                    repeat_delay=1000)
    return ani
    

#%%
a=RunState(100,1000,20,20,5,0.4,0.5,4,20)
print(a[3],a[4])
#%%
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
#%%
b=RunState(100,4000,100,100,5,0.4,0.6,4,20)
x=[]
y=[]
for i in range(len(b[1])):
    x.append(b[1][i][0])
    y.append(b[1][i][1])

fig=plt.figure()

ims=[]
for i in range(len(b[0])):
    im=plt.scatter(x,y,c=b[0][i],edgecolors='r',cmap=plt.cm.binary)
    ims.append([im])
plt.colorbar()

ani = animation.ArtistAnimation(fig, ims, interval=100, 
                                repeat_delay=1000)
#%%
N = 4000
x_size = 100
y_size = 100
cc=RunState(100,N,x_size,y_size,5,0.4,0.6,4,20)

x=[]
y=[]
for i in range(len(cc[1])):
    x.append(cc[1][i][0])
    y.append(cc[1][i][1])

fig=plt.figure()

ims=[]
for i in range(len(cc[0])):
    im=plt.scatter(x,y,c=cc[0][i],edgecolors='r',cmap=plt.cm.binary)
    ims.append([im])
plt.colorbar()

ani = animation.ArtistAnimation(fig, ims, interval=100, 
                                repeat_delay=1000)

#%%
x_grid_size = 10
y_grid_size = 10
N_grid = x_grid_size*y_grid_size

grid_coor = []
for i in range(int(x_grid_size)):
    for j in range(int(y_grid_size)):
        grid_coor.append([i,j])

grid_container = []
for i in range(len(grid_coor)):
    grid_container.append([])
for i in range(len(cc[1])):
    grid_coor_state = round_down(cc[1][i][0]/(x_size/x_grid_size)), round_down(cc[1][i][1]/(y_size/y_grid_size)) 
    print(int(grid_coor_state[0]*(x_grid_size) + grid_coor_state[1]),grid_coor_state)
    grid_container[int(grid_coor_state[0]*(x_grid_size) + grid_coor_state[1] )].append(i)

fig = plt.figure()
ims=[]
for i in range(len(cc[0])):
    grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
    for cell in range(len(grid_container)):
        sum = 0
        for node in range(len(grid_container[cell])):
            sum = sum + cc[0][i][grid_container[cell][node]]
        grid_sum[grid_coor[cell][1]][grid_coor[cell][0]] = sum
    
    
    im=plt.imshow(grid_sum,interpolation='none',cmap=plt.cm.binary,animated=True)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims, interval=100, 
                                repeat_delay=1000)
#%%testing the new function for animation of nodes
A=RunState(100,1000,20,20,5,0.4,0.5,4,20)
MovieNodes(A)
#%%testing the new function for animation of pixels
c=RunState(100,4000,100,100,5,0.4,0.6,4,20)
MoviePixels(c,100,100,10,10)