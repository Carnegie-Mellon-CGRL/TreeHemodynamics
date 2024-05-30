
import pickle
import vtk

def main(path, suffix):

    radii = pickle.load(open(path + 'radii' + suffix + '.p','rb'))
    conn = pickle.load(open(path + 'connectivity' + suffix + '.p','rb'))
    xyz = pickle.load(open(path + 'verticies' + suffix + '.p','rb'))
    pressure = pickle.load(open(path + 'pressure' + suffix + '.p','rb'))
    flow = pickle.load(open(path + 'flow' + suffix + '.p','rb'))
    wss = pickle.load(open(path + 'wss' + suffix + '.p','rb'))
    
    g = vtk.vtkMutableUndirectedGraph()
    print(len(conn))
    print(len(xyz))
    points = vtk.vtkPoints()
    for i in xyz:
    	i[0] = i[0]*10.02
    	i[1] = i[1]*10.02
    	i[2] = i[2]*10.02
    	points.InsertNextPoint(i)
    	g.AddVertex()
    
    for edge in conn:
    	g.AddEdge(edge[0],edge[1])
    
    g.SetPoints(points)
    
    graphToPolyData = vtk.vtkGraphToPolyData()
    graphToPolyData.SetInputData(g)
    graphToPolyData.Update()
    cl = graphToPolyData.GetOutput()
    
    vtk_radii = vtk.vtkDoubleArray()
    for i in radii:
    	vtk_radii.InsertNextValue(i)
    vtk_radii.SetName('Radius (um)')
    cl.GetPointData().AddArray(vtk_radii)
    
    vtk_pressure = vtk.vtkDoubleArray()
    for p in pressure:
     	vtk_pressure.InsertNextValue(p)
    vtk_pressure.SetName('Pressure (mmHg)')
    cl.GetCellData().AddArray(vtk_pressure)
    
    vtk_flow = vtk.vtkDoubleArray()
    for f in flow:
    	vtk_flow.InsertNextValue(f * 60)
    vtk_flow.SetName('Flow (mL/min)')
    cl.GetCellData().AddArray(vtk_flow)
    
    vtk_wss = vtk.vtkDoubleArray()
    for w in wss:
     	vtk_wss.InsertNextValue(w)
    vtk_wss.SetName('WSS (dyne/cm^2)')
    cl.GetCellData().AddArray(vtk_wss)
    
    vtk_id = vtk.vtkDoubleArray()
    for i in range(0,cl.GetNumberOfPoints()):
    	vtk_id.InsertNextValue(i)
    vtk_id.SetName('Id')
    cl.GetPointData().AddArray(vtk_id)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(graphToPolyData.GetOutput())
    #writer.SetFileName('graph_raw_new.vtp')
    #writer.SetFileName('graph_raw_pruned.vtp')
    writer.SetFileName(path + 'graph_hemo' + suffix + '.vtp')
    writer.Write()
    
    print(cl.GetPoint(0))
    
if __name__ == '__main__':
    # executed as script
    main('/home/jszafron/Documents/source/PAMorphometry/AVResults/graph_hemo/', '_reduced_test')