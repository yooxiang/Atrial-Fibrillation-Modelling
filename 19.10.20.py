# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 17:38:02 2020

@author: ahadj

Initial attempt on creating a random geometric model
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def GenerateCoorArray(N,x_size,y_size):
    pmcell=int(N/10)#number of pacemaker cells
    CoorStore = []
    for i in range(0,pmcell):
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

def distance_checker(R,i,j,CoorStore):
    d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    if d < R:
        return True
    else:
        return False

def ConnectionArray(N,x_size,y_size,CoorStore,R):
    ConnectionsStore = []
    for i in range(N):
        connections = []
        for j in range(N):
            if i !=j:
                if distance_checker(R,i,j,CoorStore) == True:
                    connections.append(j)
        ConnectionsStore.append(connections)
    return ConnectionsStore

def uniformdistrabutor(x):
    return x


def FunctionalArray(N,x_size,y_size,tau,delta,epsilon):
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

def GenerateState(N):
    Statelist=[]
    for i in range(N):
        Statelist.append(0)
    return Statelist
#%%      
def PaceMaker(StateLattice,FunctionLattice,N):
    pmcell=int(N/10)
    for i in range(pmcell):
        rand=random.random()
        if rand>FunctionLattice[1][i]:
            StateLattice[i]= FunctionLattice[0][i] + 1                
    return StateLattice

def UpdateStateLattice(N, StateLattice, FunctionLattice, Connection):
  
    Updatedlist = []
    for i in range(N):
        Updatedlist.append(0)
    for i in range(N):
        if StateLattice[i] > 0: #reduce refractory period countdown by one
            Updatedlist[i] = StateLattice[i]- 1
            continue
            
        if StateLattice[i]== 0: #here check if any neighbours are in firing state
            rand=random.random()
            for j in range(len(Connection[i])):
                NeighNode=Connection[i][j]
                if StateLattice[NeighNode] == FunctionLattice[0][NeighNode]+1:
                    if rand>FunctionLattice[1][NeighNode]:
                        Updatedlist[i]=FunctionLattice[0][i]+1
    return Updatedlist

def RunState(TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T):
    Coordinates = GenerateCoorArray(N,x_size,y_size)
    Connections = ConnectionArray(N,x_size,y_size,Coordinates,r)
    FunctionListR = FunctionalArray(N,x_size,y_size,tau,delta,epsilon)
    StateLattice = GenerateState(N)
    
    LatticeStore = []

    for i in range(TimeSteps):
        
        if i == 0 or i % T == 0:
            StateLattice = PaceMaker(StateLattice,FunctionListR,N)
            LatticeStore.append(StateLattice)
        else:
            StateLattice = UpdateStateLattice(N,StateLattice,FunctionListR,Connections )
            LatticeStore.append(StateLattice)
    return LatticeStore, Coordinates, Connections

#%%
a=RunState(100,1000,100,100,5,0.3,0.3,7,20)
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

