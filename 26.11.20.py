# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 02:19:31 2020

@author: ahadj
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import math
import timeit
import copy

def round_down(n, decimals=0):#used to identify in which cell a node belongs to
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def GenerateCoorArray(N,x_size,y_size):#prepares a list with the coordinates of all nodes
    pmcell=int(N/10)#number of pacemaker cells
    CoorStore =[]
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
        d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    elif abs(diff)>R:    
        if CoorStore[i][1]+R>y_size:#sets upper BC
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]-y_size - CoorStore[j][1]) **2 )**0.5
        elif CoorStore[i][1]-R<0:#sets lower BC
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]+y_size - CoorStore[j][1]) **2 )**0.5
        else:
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    if d <= R:
        return True
    else:
        return False       

def ConnectionArray(N,x_size,y_size,CoorStore,R,t_memory):
    #a list containing all the connections of the ith node in list
    t_memory_array = np.zeros(t_memory)
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
                    ChargeFlow.append(np.copy(t_memory_array))
        ConnectionsStore.append(connections)
        ChargeFlowStore.append(ChargeFlow)
        
    ChargeFlowOUTstore = copy.deepcopy(ChargeFlowStore)
    ChargeFlowINstore = copy.deepcopy(ChargeFlowStore)
    return ConnectionsStore, ChargeFlowINstore, ChargeFlowOUTstore

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
    return [0]*N
#%%      
def PaceMaker(StateListP,FunctionListP,N):#excites pacemaker cells every T time steps
    pmcell=int(N/10)
    for i in range(pmcell):
        rand=random.random()
        if rand>FunctionListP[1][i]:
            StateListP[i]= FunctionListP[0][i] + 1                
    return StateListP

def UpdateStateLattice(N, StateListU, FunctionListU, Connection,ChargeFlowIn,ChargeFlowOut):
#updates the state of each node
  
    ChargeFlowIN=copy.deepcopy(ChargeFlowIn)
    ChargeFlowOUT = copy.deepcopy(ChargeFlowOut)
    
    Updatedlist = [0]* N
    for i in range(N):
        if StateListU[i] > 0: #reduce refractory period countdown by one
            Updatedlist[i] = StateListU[i]- 1            
            for j in range(len(Connection[i])):
                NeighNode=Connection[i][j]
                ChargeFlowIn,ChargeFlowOut = ChargeFlowUpdate(ChargeFlowIN, ChargeFlowOUT,i,j,NeighNode,False,Connection)               
            #continue
        if StateListU[i]== 0: #here check if any neighbours are in firing state
            rand=random.random()
            for j in range(len(Connection[i])):
                NeighNode=Connection[i][j]
                if StateListU[NeighNode] == FunctionListU[0][NeighNode]+1:
                    if rand>FunctionListU[1][NeighNode]:
                        Updatedlist[i]=FunctionListU[0][i]+1
                        ChargeFlowIn,ChargeFlowOut = ChargeFlowUpdate(ChargeFlowIN, ChargeFlowOUT,i,j,NeighNode,True,Connection)
                else:
                    ChargeFlowIn,ChargeFlowOut = ChargeFlowUpdate(ChargeFlowIN, ChargeFlowOUT,i,j,NeighNode,False,Connection)
                    
    return Updatedlist,ChargeFlowIn, ChargeFlowOut
 
def ChargeFlowUpdate(ChargeFlowIn, ChargeFlowOut,i,j,NeighNode,state,Connection):

    if state == True:
        ChargeFlowIn[i][j] = np.insert(ChargeFlowIn[i][j],0,1) #insert the first element with updated
        ChargeFlowIn[i][j] = ChargeFlowIn[i][j][:-1] #remove the last element ie getting rid of the oldest value
        
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = np.insert(ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)],0,1)
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)][:-1]
        
    if state == False:
        ChargeFlowIn[i][j] = np.insert(ChargeFlowIn[i][j],0,0) #replace the first element with updated
        ChargeFlowIn[i][j] = ChargeFlowIn[i][j][:-1] #remove the last element ie getting rid of the oldest value
        
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = np.insert(ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)],0,0)
        ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)] = ChargeFlowOut[NeighNode][Connection[NeighNode].index(i)][:-1]
        
    return ChargeFlowIn,ChargeFlowOut

#%%
def RunState(TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T,t_memory):
    #runs the model for a specified number of timesteps
    #and returns a list with all timesteps
    SetupS=timeit.default_timer()
    Coordinates = GenerateCoorArray(N,x_size,y_size)
    Connections, ChargeFlowIn, ChargeFlowOut = ConnectionArray(N,x_size,y_size,Coordinates,r,t_memory)
    FunctionListR = FunctionalArray(N,x_size,y_size,tau,delta,epsilon)
    StateListR = GenerateState(N)
    SetupE=timeit.default_timer()
    
    StateStore = []
    
    ChargeFlowINAll=[]
    ChargeFlowOUTALL = []
    
    RunS=timeit.default_timer()
    for i in range(TimeSteps):
        print(i)
        if i == 0 or i % T == 0:          
            StateListR = PaceMaker(StateListR,FunctionListR,N)
            StateListR, ChargeFlowIn, ChargeFlowOut = UpdateStateLattice(N,StateListR,FunctionListR,Connections,ChargeFlowIn,ChargeFlowOut)
            StateStore.append(StateListR)
            ChargeFlowINAll.append(ChargeFlowIn)
            ChargeFlowOUTALL.append(ChargeFlowOut)
 
        else:
            StateListR, ChargeFlowIn, ChargeFlowOut = UpdateStateLattice(N,StateListR,FunctionListR,Connections,ChargeFlowIn,ChargeFlowOut)
            StateStore.append(StateListR)
            ChargeFlowINAll.append(ChargeFlowIn)
            ChargeFlowOUTALL.append(ChargeFlowOut)
             
    RunE=timeit.default_timer()    
    TimeS=SetupE-SetupS
    TimeR=RunE-RunS
    data=[TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T]
    return StateStore, Coordinates, Connections,TimeS,TimeR, ChargeFlowINAll, ChargeFlowOUTALL, TimeSteps,data
 
def Resultant_Vectors(ChargeFlowINALL,ChargeFlowOUTALL, Connections, Coordinates,TimeSteps,R,y_size,outv=False,inv=False):
    ChargeFlowINsingle = copy.deepcopy(ChargeFlowINALL)
    ChargeFlowOUTsingle = copy.deepcopy(ChargeFlowOUTALL)    
    ChargeFlowResultant = copy.deepcopy(ChargeFlowINALL)
    
    for t in range(TimeSteps):
        for i in range(len(Connections)):         
            for j in range(len(Connections[i])): #for each time step calcualte the memory avergae
                ChargeFlowINsingle[t][i][j] = np.mean(ChargeFlowINALL[t][i][j])
                ChargeFlowOUTsingle[t][i][j] = np.mean(ChargeFlowOUTALL[t][i][j])
                            
    for t in range(TimeSteps):
        for i in range(len(Connections)):
            INvector = [0,0]
            OUTvector = [0,0]
            avgcheck=False
            #so that the resulting vector is the average
            if outv==False and inv==False and t!=0:
                totchargeIn=sum(ChargeFlowINsingle[t-1][i])
                avgcheck=True
            else: 
                totchargeIn=sum(ChargeFlowINsingle[t][i])
            totchargeOut=sum(ChargeFlowOUTsingle[t][i])
            
            for j in range(len(Connections[i])):
                if totchargeIn!=0:                    
                    ucomIn=Coordinates[i][0] - Coordinates[Connections[i][j]][0]
                    INvector[0] += (ucomIn*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                    vcomIn=Coordinates[i][1] - Coordinates[Connections[i][j]][1]                                        
                    if abs(vcomIn)<=R:                                                                        
                        INvector[1] += (vcomIn*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                    elif abs(vcomIn)>R:
                        if vcomIn+R>y_size:                            
                            INvector[1] += ((vcomIn-y_size)*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                        elif vcomIn<0:                            
                            INvector[1] += ((vcomIn+y_size)*(ChargeFlowINsingle[t][i][j]/totchargeIn))
                        
                if totchargeOut!=0:
                    ucomOut=Coordinates[Connections[i][j]][0] - Coordinates[i][0]
                    OUTvector[0] += (ucomOut*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))
                    vcomOut=Coordinates[Connections[i][j]][1] - Coordinates[i][1]
                    if abs(vcomOut)<=R:                        
                        OUTvector[1] += (vcomOut*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))
                    elif abs(vcomOut)>R:
                        if vcomOut+R>y_size:                            
                            OUTvector[1] += ((vcomOut-y_size)*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))
                        elif vcomOut<0:
                            OUTvector[1] += ((vcomOut+y_size)*(ChargeFlowOUTsingle[t][i][j]/totchargeOut))
                #condition for taking the average between the previous time step of in vector
                if totchargeIn!=0 and avgcheck==True:                    
                    ucomIn=Coordinates[i][0] - Coordinates[Connections[i][j]][0]
                    INvector[0] += (ucomIn*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))
                    vcomIn=Coordinates[i][1] - Coordinates[Connections[i][j]][1]                                        
                    if abs(vcomIn)<=R:                                                                        
                        INvector[1] += (vcomIn*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))
                    elif abs(vcomIn)>R:
                        if vcomIn+R>y_size:                            
                            INvector[1] += ((vcomIn-y_size)*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))
                        elif vcomIn<0:                            
                            INvector[1] += ((vcomIn+y_size)*(ChargeFlowINsingle[t-1][i][j]/totchargeIn))                

            if outv==False and inv==False:
                #print('hereavg')
                ChargeFlowResultant[t][i] = [(INvector[0]+OUTvector[0])/2,(INvector[1]+OUTvector[1])/2]
            if outv==True:
                mag= np.sqrt((OUTvector[0])**2+(OUTvector[1])**2)
                if mag != 0:
                    finalvector=[OUTvector[0]/mag,OUTvector[1]/mag]
                else:
                    finalvector=[OUTvector[0],OUTvector[1]]
                ChargeFlowResultant[t][i]=finalvector
            if inv==True:
                mag= np.sqrt((INvector[0])**2+(INvector[1])**2)
                if mag != 0:
                    finalvector=[INvector[0]/mag,INvector[1]/mag]
                else:
                    finalvector=[INvector[0],INvector[1]]                
                ChargeFlowResultant[t][i]=finalvector
    return ChargeFlowResultant #this is hopefully a list arranged first by time, and then each ndoe given a vector in the form of [x,y]       
            
def MovieNodes(a):#input is list with all the lists describing the state of nodes(StateStore)
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
    ani = animation.ArtistAnimation(fig, ims, interval=500, 
                                     repeat_delay=1000)

    return ani

def Pixelation(cc,x_grid_size,y_grid_size):
    #prepares pixelation of nodes based on resolution requested
    #cc is output of RunState
    x_size=cc[8][2]
    y_size=cc[8][3]
    tau=cc[8][4]
    grid_coor = []
    for j in range(int(y_grid_size)):
        for i in range(int(x_grid_size)):
            grid_coor.append([i,j])
    grid_container = []
    timeseries=[]#contains time-series for each cell
    for i in range(len(grid_coor)):
        grid_container.append([])
        timeseries.append([])
    for i in range(len(cc[1])):
        grid_coor_state = cc[1][i][0]//(x_size/x_grid_size),cc[1][i][1]//(y_size/y_grid_size) 
        grid_container[int(grid_coor_state[1]*(x_grid_size) + grid_coor_state[0] )].append(i)
    allgridvalues=[]
    for i in range(len(cc[0])):
        grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
        for cell in range(len(grid_container)):
            sum_c = 0
            for node in range(len(grid_container[cell])):
                sum_c += cc[0][i][grid_container[cell][node]]
            grid_sum[y_grid_size-1-grid_coor[cell][1]][grid_coor[cell][0]]=sum_c
            timeseries[cell].append(sum_c)
        allgridvalues.append(grid_sum)                
    nodespc=[]#nodespercell(determining cell with max number of nodes)
    for i in range(len(grid_container)):
        nodespc.append(len(grid_container[i]))
    maxcellcolor=np.mean(nodespc)*(tau+1)#determining max value possible 
    #in grid_sum,required to set the color scale
    return allgridvalues,int(maxcellcolor) ,timeseries,grid_container,grid_coor

def MoviePixels(pixeldata):
    #input is output of Pixelation
    Allgridvalues=pixeldata[0]
    fig = plt.figure()
    ims=[]    
    for i in range(len(Allgridvalues)):
        im=plt.imshow(Allgridvalues[i],interpolation='none',cmap=plt.cm.binary,vmin=0,vmax=pixeldata[1],animated=True)
        if i==0:
            fig.colorbar(im)
        ims.append([im])
    plt.title('Pixelated Grid')
    ani = animation.ArtistAnimation(fig, ims, interval=500, 
                                    repeat_delay=1000)
    return ani

def PixelatedVectors(cc,cv,x_grid_size,y_grid_size):
    #prepares exact coarse graining
    #cc is output of Runstate
    #cv is output of resultant vector
    x_size=cc[8][2]
    y_size=cc[8][3]
    grid_coor = []#cotains coordinates for each pixel
    for j in range(int(y_grid_size)):
        for i in range(int(x_grid_size)):
            grid_coor.append([i,j])
    grid_container = []#contains nodes in each pixel
    for i in range(len(grid_coor)):
        grid_container.append([])
    for i in range(len(cc[1])):
        grid_coor_state = cc[1][i][0]//(x_size/x_grid_size),cc[1][i][1]//(y_size/y_grid_size) 
        grid_container[int(grid_coor_state[1]*(y_grid_size) + grid_coor_state[0] )].append(i)
    
    gridvectorTot=[]
    for t in range(len(cv)):
        gridvector_t=[]
        for cell in range(len(grid_container)):
            cellvector=[0,0]
            no_nodes=len(grid_container[cell])
            for i in range(no_nodes):
                cellvector[0]+=cv[t][grid_container[cell][i]][0]/no_nodes
                cellvector[1]+=cv[t][grid_container[cell][i]][1]/no_nodes
            cellvector_mag=np.sqrt((cellvector[0])**2+(cellvector[1])**2)
            if cellvector_mag!=0:
                cellvector_norm=[cellvector[0]/cellvector_mag,cellvector[1]/cellvector_mag]
            else:
                cellvector_norm=cellvector
            gridvector_t.append(cellvector_norm)
        gridvectorTot.append(gridvector_t)                       
    return gridvectorTot,grid_coor
#%%
def divergence(vector,Coordiantes,x_dim,y_dim): #x_dim and y_dim are nunber of cells, not the true length which is captured in the coordinates
    
    def internal_point(index): # this is for all the points apart from the edges on the left and right
        #vector = vector_field[index]
        y_coor =(index//x_dim)        
        dFx = (vector[index+1][0] - vector[index-1][0])/(Coordiantes[index+1][0] - Coordiantes[index-1][0])
                
        if y_coor == 0:
            dFy = (vector[index+x_dim][1] - vector[index+(x_dim*(y_dim -1))][1])/(2*(Coordiantes[index+x_dim][1] - Coordiantes[index][1]))
        if y_coor == y_dim - 1:   
            dFy = (vector[index-(x_dim*(y_dim -1))][1] - vector[index - x_dim][1])/(2*(Coordiantes[index][1] - Coordiantes[index-x_dim][1]))
        else:
            dFy = (vector[index+x_dim][1] - vector[index - x_dim][1])/(Coordiantes[index+x_dim][1] - Coordiantes[index-x_dim][1])         
        return dFx + dFy   
    
    def edge_point(index): #this is for all the points on the right and left ie x = 0 or x= y_di
        #here a central difference is not used
        #vector = vector_field[index]
        x_coor = index % y_dim 
        y_coor =(index//x_dim)
        
        if x_coor == 0:          
            dFx = (vector[index+1][0] - vector[index][0])/(Coordiantes[index+1][0] - Coordiantes[index][0])            
        if x_coor == y_dim - 1:
            dFx = (vector[index][0] - vector[index-1][0])/(Coordiantes[index][0] - Coordiantes[index-1][0])            
        if y_coor == 0:        
            dFy = (vector[index+x_dim][1] - vector[index+(x_dim*(y_dim -1))][1])/(2*(Coordiantes[index+x_dim][1] - Coordiantes[index][1]))
        if y_coor == y_dim - 1:
            dFy = (vector[index-(x_dim*(y_dim -1))][1] - vector[index - x_dim][1])/(2*(Coordiantes[index][1] - Coordiantes[index-x_dim][1]))            
        if y_coor > 0 and y_coor < y_dim - 1:
            dFy = (vector[index+x_dim][1] - vector[index - x_dim][1])/(Coordiantes[index+x_dim][1] - Coordiantes[index-x_dim][1])
        
        return dFx + dFy

    div_store = []
    for index in range(x_dim*y_dim):
        x_coor = index % y_dim 

        if x_coor > 0 and x_coor < y_dim -1:
            div = internal_point(index)
        else:
            div = edge_point(index)
        div_store.append(div)
    
    return div_store,Coordiantes
#following function creates vector movies
def VectorMovie(vectordata,points,frame):
    #vectordata must be the vector of either nodes or pixels
    #with their respective points
    #if frame==None then it returns the whole movie otherwise specify
    #the frame you need to visualise
    X=[i[0] for i in points]
    Y=[i[1] for i in points]
    U=[0]*len(points)
    V=[0]*len(points)
    
    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X,Y,U, V, pivot='tail', angles='xy', scale_units='xy',scale=1)
    
    def update_quiver(num,Q):
        U=[0]*len(points)
        V=[0]*len(points)
        Q.set_UVC(U,V)
        for i in range(len(points)):
            U[i]=vectordata[num][i][0]
            V[i]=vectordata[num][i][1]
        colors = np.arctan(np.array(V),np.array(U))
        Q.set_UVC(U,V,colors)
        return Q,
    
    if frame==None:
        anim1 = animation.FuncAnimation(fig, update_quiver,frames=len(vectordata),fargs=(Q,), interval=500, blit=True)
    else:
        anim1=update_quiver(frame,Q)
    fig.tight_layout()
    
    return anim1
#function that returns movie of divergence
def DivMovie(vectordata,coord,frame_no):
    #input vectordata is the vectors of each pixel
    #coord is the coordinates of the pixels
    #if frame_no==None then it returns a Movie
    x_s=max([i[0] for i in coord])+1#sets x length of grid
    y_s=max([i[1] for i in coord])+1#sets y length of grid
    div_all=[]#list with divergence data for all times
    div_max=[]
    div_min=[]
    for i in range(len(vectordata)):
        div_loop=divergence(vectordata[i],coord,y_s,x_s)
        div_all.append(div_loop[0])
        div_max.append(max(div_loop[0]))
        div_min.append(min(div_loop[0]))
    #frame_all contains the frames for all time steps
    frame_all=[]
    for time in range(len(div_all)):
        frame_i=np.zeros([y_s,x_s])
        for cell in range(len(coord)):
            frame_i[y_s-1-coord[cell][1]][coord[cell][0]]=div_all[time][cell]
        frame_all.append(frame_i)
        
    fig = plt.figure() 
    
    if frame_no==None:
        ims=[]    
        for i in range(len(frame_all)):
            im=plt.imshow(frame_all[i],interpolation='none',cmap='jet',vmin=min(div_min),vmax=max(div_max),animated=True)
            if i==0:
                fig.colorbar(im)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=500,repeat_delay=1000)
    
    else:
        ani=plt.imshow(frame_all[frame_no],interpolation='none',cmap='jet',vmin=min(div_min),vmax=max(div_max),animated=True)
        fig.colorbar(ani)
    plt.title('Pixelated Grid')
    
    return ani
#%%run to get data
x17 = RunState(50,2000,65,65,5,1,0.35,5,20,1)
#%%visualise microscopic data
MovieNodes(x17)
#%%obtain the vectors for each node
Vecx17=Resultant_Vectors(x17[5],x17[6],x17[2],x17[1],x17[7],5,65,outv=True)
#%%data for pixelated vectors
VectorPixels=PixelatedVectors(x17,Vecx17,9,9)
#%%
def ReentrantDot(vectordata,points,xdim,ydim):
    avgdot_all=[]
    for time in range(len(vectordata)):
        avgdot_time=[]
        for i in range(len(vectordata[time])):
            x_loc=i%xdim#find x coord. of respective pixel
            y_loc=i//xdim#finds y coord. of respective pixel
            surrvec=[]#surrounding vectors
            
            if x_loc==(xdim-1):#right BC
                right=0
            else:
                right=sum([vectordata[time][i][j]*vectordata[time][i+1][j] for j in range(len(vectordata[time][i]))])
                surrvec.append(right)
            if x_loc!=0:#left BC
                left=sum([vectordata[time][i][j]*vectordata[time][i-1][j] for j in range(len(vectordata[time][i]))])
                surrvec.append(left)
            else:
                left=0
            if y_loc==(ydim-1):# up BC
                up=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)][j] for j in range(len(vectordata[time][i]))])
            else:
                up=sum([vectordata[time][i][j]*vectordata[time][i+xdim][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(up)
            if y_loc==0:#down BC
                down=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)][j] for j in range(len(vectordata[time][i]))])
            else:    
                down=sum([vectordata[time][i][j]*vectordata[time][i-xdim][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(down)
            #below are diagonal terms
            if x_loc==xdim-1:#up right diag
                uprightdiag=0
            else:
                if y_loc==(ydim-1):
                    uprightdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)+1][j] for j in range(len(vectordata[time][i]))])
                else:
                    uprightdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim+1][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(uprightdiag)
            if x_loc==(xdim-1):#down right diag
                downrightdiag=0
            else:
                if y_loc==0:
                    downrightdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)+1][j] for j in range(len(vectordata[time][i]))])
                else:
                    downrightdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+1][j] for j in range(len(vectordata[time][i]))])                    
            surrvec.append(downrightdiag)
            if x_loc==0:#up left diag
                upleftdiag=0
            else:
                if y_loc==(ydim-1):
                    upleftdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)-1][j] for j in range(len(vectordata[time][i]))])
                else:
                    upleftdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-1][j] for j in range(len(vectordata[time][i]))])            
            surrvec.append(upleftdiag)
            if x_loc==0:#down left diag
                downleftdiag=0
            else:
                if y_loc==0:
                    downleftdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)-1][j] for j in range(len(vectordata[time][i]))])
                else:
                    downleftdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim-1][j] for j in range(len(vectordata[time][i]))]) 
            surrvec.append(downleftdiag)
            
            surrvecN=len(surrvec)
            if surrvecN==0:
                avgdot=None
            else:
                avgdot=sum(surrvec)/surrvecN
            avgdot_time.append(avgdot)
        
        avgdot_all.append(avgdot_time)
    
    return avgdot_all