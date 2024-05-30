#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:59:06 2024

@author: jszafron
"""
from datetime import date
from graph_tool.all import *
import shutil
import numpy as np
import math
import random
import pickle
import reduce_graph_helper
import matplotlib.pyplot as plt
import pysvzerod
import json
import pdb

path = '/home/jszafron/Documents/source/TreeHemodynamics/'
directory = str(path)

import os
import sys
file_dir = os.path.dirname(directory)
sys.path.append(file_dir)

import graphToCenterline

#img resolution 10 um/pix
voxel_2_um = 10 #um/vx scaling for uCT image
um_2_cm = 1E-4 #conversion cm/mm

#ROM_simulation parameters
input_file_name = 'AV21_5_test.json'
model_name = 'AV21_5_test'

# input_file_name = 'AV443_3_test.json'
# model_name = 'AV443_3_test'

#Vessel elements
vessel_elems = ['R_poiseuille']
types_vessel_elems = ['R_poiseuille', 'C']

#Flow parameters
CO = 16.90 / 60  #mL/s
inflow_t = [0, 1]
inflow_q = [CO, CO]

#Time parameters
dt = 0.01
n_step = 10
t_cycle = 1

#Fluid paramaters
density = 1.00
viscosity = 0.04
#Outlet BC parameters
outlet_R = [0]
Pd = [4.3 * 1333.33]

inp = {'simulation_parameters': {},
        'boundary_conditions': [],
        'junctions': [],
        'vessels': []}

#Simulation Parameters
# general
inp['simulation_parameters']['model_name'] = model_name
# time
inp['simulation_parameters']['number_of_time_pts_per_cardiac_cycle'] = int(t_cycle / dt)
inp['simulation_parameters']['number_of_cardiac_cycles'] = int(n_step / t_cycle * dt) + 1
# fluid
inp['simulation_parameters']['density'] = density
inp['simulation_parameters']['viscosity'] = viscosity

#Load bifurcating graph
g_prune = load_graph(path + 'graph_pruned_reduced.gt')

#Trim graph for testing
# keep_vertices = [60, 63, 646, 753, 772, 775, 806, 818, 837, 844, 845, 882, 889, \
# 1051, 1098, 1288, 1516, 1540, 1547, 1611, 13642, 13667, 13755, 13761, 13767, \
# 13778, 13782, 13942] #for MPA
# #keep_vertices = [60, 63, 844, 882, 13642, 13667, 13755, 13942] #for MPA
# remove_ind = []
# for v in g_prune.iter_vertices():
#     if v not in keep_vertices:
#         remove_ind.append(v)
        
# g_prune.remove_vertex(remove_ind)

#Optional code to remove 
#remove vertexes smaller than threshold
# min_vert_size = 15 # in px, ~0.03 mm / pix
# remove_ind = []
# for v in g_prune.iter_vertices():
#     if g_prune.vp.radii[v] < min_vert_size:
#         remove_ind.append(v)

# g_prune.remove_vertex(remove_ind)
# g_prune = extract_largest_component(g_prune, None, True)

vertex_coords = []
radii = []
conn = []
for v in g_prune.iter_vertices():
    vertex_coords.append(g_prune.vp.coordinates[v])
    radii.append(g_prune.vp.radii[v] * voxel_2_um)

conn_save = directory +  "connectivity_reduced_test.p"
vert_save = directory +  "verticies_reduced_test.p"
rad_save = directory +  "radii_reduced_test.p"

conn = g_prune.get_edges()

pickle.dump(conn, open( conn_save, "wb" ) )
pickle.dump(vertex_coords, open( vert_save, "wb" ) )
pickle.dump(radii, open( rad_save, "wb" ) )

g_prune.save(path + "graph_reduced_test.gt", fmt="gt")

#reduce_graph_helper.main(path, '_test')

graphToCenterline.main(path, '_reduced_test')

g_prune = load_graph(path + 'graph_reduced_test.gt')

#Reduce graph
#reduce_graph_helper.main(path)
#Generate intermediate vtk graph


#Vertices are junctions
#find input vertex ind
curr_max_rad = 0
curr_max_ind = 0
num_out = 0
for v in g_prune.iter_vertices():
    if g_prune.get_total_degrees([v]) == 1:
        num_out += 1
        if g_prune.vp.radii[v] > curr_max_rad:
            curr_max_rad = g_prune.vp.radii[v]
            curr_max_ind = v
input_vertex_ind = curr_max_ind
in_vertex = g_prune.vertex(input_vertex_ind)

joint_count = 0
distances = shortest_distance(g_prune, in_vertex)
distance_arr = distances.get_array()
for v in g_prune.iter_vertices():
    if g_prune.get_total_degrees([v]) > 1:
        
        nb_vert = g_prune.get_all_neighbors(v)
        
        #Set the inlet to the segment with the shortest path
        inlet_vert = nb_vert[np.argmin(distance_arr[nb_vert])]
        inlet_edge = g_prune.edge(inlet_vert, v)
        inlet_edge_ind = g_prune.edge_index[inlet_edge]
        
        junction = {'junction_name': 'J' + str(joint_count),
                    'junction_type': 'NORMAL_JUNCTION',
                    'inlet_vessels': [int(inlet_edge_ind)],
                    'outlet_vessels': []}
        
        #Gather the outlet segments
        outlet_vert = np.delete(nb_vert, np.argmin(distance_arr[nb_vert]))
        num_outlet = len(outlet_vert)
        outlet_edge = []
        outlet_edge_ind = []
        for ov in range(num_outlet):
            curr_outlet_edge = g_prune.edge(v, outlet_vert[ov])
            curr_outlet_edge_ind = g_prune.edge_index[curr_outlet_edge]
            junction['outlet_vessels'] += [int(curr_outlet_edge_ind)]
        
        inp['junctions'] += [junction]
        joint_count += 1 #increment the joint #

#Edges are vessels
edge_count = 0
outlet_count = 1
zeroD_name = []
out_res_vals = []
for e in g_prune.iter_edges():
    
    curr_edge = g_prune.edge(e[0], e[1])
    edge_ind = g_prune.edge_index[curr_edge]

    coord_v1 = g_prune.vp.coordinates[e[0]]
    coord_v2 = g_prune.vp.coordinates[e[1]]
    length = ( (coord_v1[0] - coord_v2[0])**2 + (coord_v1[1] - coord_v2[1])**2 + (coord_v1[2] - coord_v2[2])**2 )**(1 / 2) * voxel_2_um * um_2_cm
    
    #mean of nodal radii
    rn1 = g_prune.vp.radii[e[0]]
    rn2 = g_prune.vp.radii[e[1]]
    r_avg = (rn1 + rn2)/2 * voxel_2_um * um_2_cm
    R_p = 8 * viscosity * length / (np.pi * r_avg**4)
    
    vessel = {'vessel_id': int(edge_ind),
              'vessel_name': 'branch' + str(edge_count) + '_seg' + str(0),
              'vessel_length': length,
              'zero_d_element_type': 'BloodVessel',
              'zero_d_element_values': {}}
    
    #Check if inlet and print info for quantify morphometry
    # if (e[0] == in_vertex or e[1] == in_vertex):
    #     print("Group"+ str(edge_count))
    #     print(g_prune.vp.radii[e[0]])
    #     print(g_prune.vp.radii[e[1]])
    # if rn1 > 55 or rn2 > 55:
    #     print("-----")
    #     print(rn1)
    #     print(length)
    #     print(r_avg)
    #     print("-----")
    
    # zerod values
    # for zerod in model['branches'].keys():
    #     vessel['zero_d_element_values'][zerod] = model['branches'][zerod][branch][i]
    # inlet bc
    # print("-----------")
    # print("Current Edge: " +str(edge_ind))
    # print("Vert 1 degree: " + str(g_prune.get_total_degrees([e[0]])[0]))
    # print("Vert 2 degree: " + str(g_prune.get_total_degrees([e[1]])[0]))
    # print("Cond 1 degree: " + str((g_prune.get_total_degrees([e[0]])[0] == 1 or g_prune.get_total_degrees([e[1]])[0] == 1)))
    # print("Cond 2 degree: " + str(e[0] != in_vertex and e[1] != in_vertex))
    # print("-----------")
    
    if 'R_poiseuille' in vessel_elems:
        vessel['zero_d_element_values']['R_poiseuille'] = R_p
    
    if (e[0] == in_vertex or e[1] == in_vertex):
        vessel['boundary_conditions'] = {'inlet': 'INFLOW'}
    # outlet bc
    if (g_prune.get_total_degrees([e[0]])[0] == 1 or g_prune.get_total_degrees([e[1]])[0] == 1) \
        and (e[0] != in_vertex and e[1] != in_vertex):
        bc_str = 'OUT' + str(outlet_count)
        vessel['boundary_conditions'] = {'outlet': bc_str}
        
        #Find inlet distance down tree
        if g_prune.get_total_degrees([e[0]])[0] == 1:
            res_calc = 1E23 / float(distance_arr[e[0]])**10
            out_res_vals.append(res_calc)
        else:
            res_calc = 1E23 / float(distance_arr[e[1]])**10
            out_res_vals.append(res_calc)

        outlet_count += 1
    inp['vessels'] += [vessel]
    
    zeroD_name.append('branch' + str(edge_count) + '_seg' + str(edge_count))
    
    edge_count += 1
    
g_prune.edge_properties["zeroD_name"] = g_prune.new_edge_property('string', zeroD_name)
# for e in g_prune.iter_edges():
#     curr_edge = g_prune.edge(e[0], e[1])
#     print(g_prune.edge_index[curr_edge])
#     print(g_prune.ep.zeroD_name[curr_edge])

#Inlet bc
inflow = {'bc_name': 'INFLOW',
          'bc_type': 'FLOW',
          'bc_values': {'t': inflow_t, 'Q': inflow_q}}
inp['boundary_conditions'] += [inflow]

#Outlet boundary conditions
for out in range(num_out - 1):
    bc_type = "RESISTANCE"
    bc_str = 'OUT' + str(out + 1)
    bc_val = [out_res_vals[out], Pd[0]] #[outlet_R[out], Pd[out]] 

    outflow = {'bc_name': bc_str,
               'bc_type': bc_type.upper(),
               'bc_values': {}}
    
    seq = ['R', 'Pd']
    for name, val in zip(seq, bc_val):
        outflow['bc_values'][name] = float(val)
        
    # if bc_type == OutflowBoundaryConditionType.RCR:
    #     seq = ['Rp', 'C', 'Rd', 'Pd']
    #     for name, val in zip(seq, bc_val):
    #         outflow['bc_values'][name] = float(val)
    #elif bc_type == OutflowBoundaryConditionType.RESISTANCE:
        
    # elif bc_type == OutflowBoundaryConditionType.CORONARY:
    #     for name, val in bc_val['var'].items():
    #         outflow['bc_values'][name] = val
    #     outflow['bc_values']['t'] = bc_val['time']
    #     outflow['bc_values']['Pim'] = bc_val['pressure']
    #     outflow['bc_values']['Pv'] = 0.0
    inp['boundary_conditions'] += [outflow]
    

g_prune.save(path + "graph_zeroDhemo.gt", fmt="gt")

# write to file
file_name = os.path.join(path, input_file_name)
with open(file_name, 'w') as file:
    json.dump(inp, file, indent=4, sort_keys=True)
