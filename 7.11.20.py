# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:35:47 2020

@author: ahadj
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
from tqdm import tqdm
import copy

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
    ChargeFlowStore = []
    for i in range(N):
        connections = []
        ChargeFlow = []
        nodecoor=CoorStore[i]
        ll=nodecoor[0]-R
        rl=nodecoor[0]+R
        for j in range(N):
            if i !=j and ll<=CoorStore[j][0]<=rl:    
                if distance_checker(R,i,j,CoorStore,y_size) == True:
                    connections.append(j)
                    ChargeFlow.append(0)
        ConnectionsStore.append(connections)
        ChargeFlowStore.append(ChargeFlow)
    return ConnectionsStore, ChargeFlowStore

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

def UpdateStateLattice(N, StateListU, FunctionListU, Connection,ChargeFlow):
#updates the state of each node  
    ChargeFlowI=copy.deepcopy(ChargeFlow)
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
                        ChargeFlowI[i][j] = 1
    return Updatedlist,ChargeFlowI

def RunState(TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T):
    #runs the model for a specified number of timesteps
    #and returns a list with all timesteps
    SetupS=timeit.timeit()
    Coordinates = GenerateCoorArray(N,x_size,y_size)
    Connections, ChargeFlow = ConnectionArray(N,x_size,y_size,Coordinates,r)
    FunctionListR = FunctionalArray(N,x_size,y_size,tau,delta,epsilon)
    StateListR = GenerateState(N)
    SetupE=timeit.timeit()
    
    StateStore = []
    ChargeFlowAll=[]
    RunS=timeit.default_timer()
    for i in range(TimeSteps):        
        if i == 0 or i % T == 0:
            StateListR = PaceMaker(StateListR,FunctionListR,N)
            StateStore.append(StateListR)
            ChargeFlowAll.append(ChargeFlow)
        else:
            StateListR, ChargeFlow= UpdateStateLattice(N,StateListR,FunctionListR,Connections,ChargeFlow)
            StateStore.append(StateListR)
            ChargeFlowAll.append(ChargeFlow)
    RunE=timeit.default_timer()    
    TimeS=SetupE-SetupS
    TimeR=RunE-RunS
   
    return StateStore, Coordinates, Connections,TimeS,TimeR, ChargeFlowAll

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

def Arrow(Coordinates, ChargeFlow, Connections,max_charge):
#    for i in range(len(Coordinates)):
#        plt.scatter(Coordinates[i][0],Coordinates[i][1])
    x=[]
    y=[]
    u=[]
    v=[]
    for i in range(len(Coordinates)):
        allzero = True
        for j in range(len(ChargeFlow[i])):
            if ChargeFlow[i][j] != 0:
                allzero = False
        if allzero == False:
            ArrowEnd = Coordinates[i][0],Coordinates[i][1]
            ArrowStart = Coordinates[Connections[i][ChargeFlow[i].index(max(ChargeFlow[i]))]]
#            Width = max(ChargeFlow[i])/max_charge
            x.append(ArrowStart[0])
            y.append(ArrowStart[1])
            u.append(ArrowEnd[0]-ArrowStart[0])
            v.append(ArrowEnd[1]-ArrowStart[1])
#            aa=plt.quiver(ArrowStart[0],ArrowStart[1],ArrowEnd[0]-ArrowStart[0],ArrowEnd[1]-ArrowStart[1])
#    plt.xlim(0,30)
#    plt.ylim(0,30)
    plt.show()
    return aaa
#%%
x = RunState(200,200,64,64,10,0.05,0.05,20,10)
Coordinates = x[1]
ChargeFlow = x[5]
Connections = x[2]
max_charge = 40

Arrow(Coordinates, ChargeFlow, Connections,max_charge)
#%%
trial1=RunState(100,8000,128,128,5,1,0.2,4,20)
MovieNodes(trial1)
#writergif = animation.PillowWriter(fps=100)
#MovieNodes(trial1).save('trial1.gif',writer=writergif)
#%%
x = RunState(100,20,16,16,5,1,0.2,8,20)
Coordinates = x[1]
ChargeFlow = x[5]
Connections = x[2]
MovieNodes(x)
#%%
X=[]
Y=[]
U=[]
V=[]
for i in range(len(Coordinates)):
    X.append(Coordinates[i][0])
    Y.append(Coordinates[i][1])
    U.append(0)
    V.append(0)
fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='mid', angles='xy', scale_units='xy',scale=1)

def update_quiver(num,Q):
    for i in range(len(Coordinates)):
        allzero = True
        for j in range(len(ChargeFlow[num][i])):
            if ChargeFlow[num][i][j] != 0:
                    allzero = False
            if allzero == False:
                ArrowEnd = Coordinates[i][0],Coordinates[i][1]
                ArrowStart = Coordinates[Connections[i][ChargeFlow[num][i].index(max(ChargeFlow[num][i]))]]
                location=ChargeFlow[num][i].index(max(ChargeFlow[num][i]))
                U[location]=ArrowEnd[0]-ArrowStart[0]
                V[location]=ArrowEnd[1]-ArrowStart[1]
    Q.set_UVC(U,V)
    return Q,

anim = animation.FuncAnimation(fig, update_quiver,frames=len(ChargeFlow),fargs=(Q,),
                               interval=100, blit=True)
fig.tight_layout()
writergif = animation.PillowWriter(fps=100)
anim.save('trial2.gif',writer=writergif)


#%%
X, Y = np.mgrid[:2*np.pi:10j,:2*np.pi:5j]
U = np.cos(X)
V = np.sin(Y)

fig, ax = plt.subplots(1,1)
Q = ax.quiver(X, Y, U, V, pivot='mid', color='r', units='inches')

ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)

def update_quiver(num, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    print(num)
    U = np.cos(X + num*0.1)
    V = np.sin(Y + num*0.1)

    Q.set_UVC(U,V)

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                               interval=50, blit=False)
fig.tight_layout()

plt.show()
