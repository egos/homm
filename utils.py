import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# import skimage
# import skimage.transform
from types import SimpleNamespace
import networkx as nx
import itertools 

class heroes():
    def __init__(self, pos, move_point):
        self.move_point_ini  = move_point
        self.move_point = 0
        self.pos = pos
        self.money = 0
        self.update()       
        
    def update(self):
        self.move_point = self.move_point_ini

def Path1(A,start, mode = 0): 
    N = len(A)
    v0 = np.array([-1,1,-N,N])
    if mode == 1 : 
        v0 = np.array([-1,1,-N,N, -N-1, -N+1,N+1,N-1])
    Dim = len(v0)
    e = 2
    A[start] = e
    a = A.reshape(-1)
    v = np.where(a == e)  

    while len(v) > 0 :
        v = np.tile(v, (Dim, 1)).T + v0
        v = v[np.where(a[v]==0)]
        v = np.unique(v)        
        e+=1
        a[v]=e
    return a.reshape((N,N))

def Path2(A,start,goal):
    N = len(A)
    v0 = np.array([-1,1,-N,N])
    v0 = np.array([-1,1,-N,N, -N-1, -N+1,N+1,N-1])
    Dim = len(v0)
    e1,e2  = A[start] , A[goal]
    a = A.reshape(-1)
    v = goal[1] + goal[0]* N
    L  = [goal]
    while e2 > 2:
        v = v + v0
        v = v[np.where(a[v] == e2-1)][0]
        a[v] = e2
        e2 -= 1
        pos = (int(np.ceil(v/N)-1),  v%N)
        L.insert(0,pos)
    return L

def env_(Tpos_ini, custom = True, size = 10):
    if custom :
        grid0 = np.genfromtxt('A.csv', delimiter=",", dtype=int).T
        size = grid0.shape[0]
        grid = grid0.copy()
        start = (3,6)
        Tpos_ini = [(5,5),(1,3),(9,3), (9,9)] 
    else : 
        grid0 = np.full((size, size), 0)
        grid0[0,:] , grid0[:,0] , grid0[:,-1], grid0[-1,:] = 1,1,1,1
        grid = grid0.copy()
        # Tpos_ini = [(3,3)]
    
    if Tpos_ini: 
        pos = tuple(zip(*Tpos_ini))
        grid[pos] = 2
        Rcount = len(pos)
    else : 
        Rcount = 0

    start = (3,6)
    H1 = heroes(start,2)
    grid[start] = 3
    env = dict(
        grid0 = grid0,
        grid = grid,
        H1 = H1,
        size =size,
        Rcount = Rcount,
        PathMode = 0,
    )
    env = SimpleNamespace(**env)
    return env

def Grid_Path(env):
    grid = env.grid.copy()
    G = nx.Graph()
    nodes = {}
    ri = 0
    for pos in zip(*np.where(grid >= 2)):
        # print(pos)
        if grid[pos] ==2:
            node = "R{}".format(ri)
            ri+=1
        if grid[pos] ==3:
            node = "H"
        nodes[node] = {'pos' : pos}

    G.add_nodes_from(nodes.items())
    G.nodes(data = True)

    it = list(itertools.combinations(G.nodes(), 2))
    for n1 , n2 in it:
        start = G.nodes[n1]['pos']
        goal  = G.nodes[n2]['pos']
        A = (grid !=0)*1
        A[goal] = 0
        A1 = Path1(A,start, env.PathMode)
        path = Path2(A1.copy() ,start, goal)
        dist = len(path)
        G.add_edge(n1, n2, dist = dist,path = path)
    # print(G)
    return G




