#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:43:19 2024

@author: jszafron
"""
from graph_tool.all import *
import pysvzerod
import json
import pdb
import matplotlib.pyplot as plt
import numpy as np

path = '/home/jszafron/Documents/source/TreeHemodynamics/'
directory = str(path)

import os
import sys
file_dir = os.path.dirname(directory)
sys.path.append(file_dir)

import graphToCenterlineHemo
import pickle

#img resolution 10 um/pix
voxel_2_um = 10 #um/vx scaling for uCT image
um_2_cm = 1E-4 #conversion cm/mm

cgsP_2_mmHg = 7.5E-4

#Time parameters
dt = 0.01
n_step = 10
t_cycle = 1
steps = (int(n_step / t_cycle * dt) + 1) * int(t_cycle / dt)

mu = 0.04

solver = pysvzerod.Solver("AV21_5_test.json")
solver.run()

result = solver.get_full_result()

g_prune = load_graph('graph_zeroDhemo.gt')
num_edges = 0
flow_in = []
pressure_avg = []
wss_all = []
for e in g_prune.iter_edges():
    curr_end_val = (num_edges + 1) * steps - 1
    curr_edge = g_prune.edge(e[0], e[1])
    edge_ind = g_prune.edge_index[curr_edge]
    
    Q_in = result["flow_in"][curr_end_val]
    flow_in.append(Q_in)
    
    pressure_in = result["pressure_in"][curr_end_val]
    pressure_out = result["pressure_in"][curr_end_val]
    p_avg_mmHg = (pressure_in + pressure_out) / 2 * cgsP_2_mmHg
    pressure_avg.append(p_avg_mmHg)
    
    rn1 = g_prune.vp.radii[e[0]]
    rn2 = g_prune.vp.radii[e[1]]
    r_avg = (rn1 + rn2)/2 * voxel_2_um * um_2_cm
    wss = 4 * mu * Q_in / (np.pi * r_avg**3)
    wss_all.append(wss)
    
    num_edges += 1

pressure_save = directory +  "pressure_reduced_test.p"
flow_save = directory +  "flow_reduced_test.p"
wss_save = directory +  "wss_reduced_test.p"

pickle.dump(pressure_avg, open( pressure_save, "wb" ) )
pickle.dump(flow_in, open( flow_save, "wb" ) )
pickle.dump(wss_all, open( wss_save, "wb" ) )

graphToCenterlineHemo.main(path, '_reduced_test')
