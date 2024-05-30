#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:48:35 2023

@author: jszafron
"""

import os
os.chdir('/home/jszafron/Documents/ClearMap2/')
from ClearMap.Environment import * 
get_ipython().run_line_magic('gui', 'qt')
from datetime import date
import shutil
import math
import pickle

def main(path, suffix):

    directory = str(path)
    
    radii = pickle.load(open(path + 'radii' + suffix + '.p','rb'))
    conn = pickle.load(open(path + 'connectivity' + suffix + '.p','rb'))
    xyz = pickle.load(open(path + 'verticies' + suffix + '.p','rb'))
    
    g_init = grp.Graph('pruned_graph')
    g_init.add_vertex(len(xyz));
    g_init.add_edge(conn)
    
    g_init.set_vertex_coordinates(xyz)
    
    verts = g_init.vertices
    g_init.add_vertex_property('radii', dtype = type(radii[0]))
    for i in range(len(verts)):
        g_init.set_vertex_radius(verts[i], radii[i])
    
    #g_init = grp.load(path + 'graph_pruned.gt')
    
    #graph_init_sk = gp.graph_to_skeleton(g_init)
    #graph_init_sk2grp = gp.graph_from_skeleton(graph_init_sk)
    #graph_init_sk2grp.set_vertex_radii(g_init.vertex_radii())
    #
    ##g_init.save(path + 'graph_pruned2.gt')
    ##g_init2 = grp.load(path + 'graph_pruned2.gt')
    #
        
    def vote(expression):
      return np.sum(expression) >= len(expression) / 1.5;
    
    graph_reduced = gp.reduce_graph(g_init, edge_length=True,
                              edge_to_edge_mappings = {'length' : np.sum},
                              vertex_to_edge_mappings={'artery_binary' : vote,
                                                       'artery_raw'    : np.max,
                                                       'radii'         : np.max},
                              edge_geometry_vertex_properties=['coordinates', 'radii', 'artery_binary', 'artery_raw'],
                              edge_geometry_edge_properties=None,
                              return_maps=False, verbose=True)
    
    # Save reduced graph
    graph_reduced.save(path + 'graph_reduced' + suffix + '.gt')
    
    verticies = graph_reduced.vertex_coordinates()
    connectivity = graph_reduced.edge_connectivity()
    radii = graph_reduced.vertex_radii()
    
    conn_save = directory +  "connectivity_reduced" + suffix + ".p"
    vert_save = directory +  "verticies_reduced" + suffix + ".p"
    rad_save = directory +  "radii_reduced" + suffix + ".p"
    
    pickle.dump(connectivity, open( conn_save, "wb" ) )
    pickle.dump(verticies, open( vert_save, "wb" ) )
    pickle.dump(radii, open( rad_save, "wb" ) )
    
if __name__ == '__main__':
    # executed as script
    main('/home/jszafron/Documents/', '_test')