
import pickle
import vtk

def main(path, suffix):

    radii = pickle.load(open(path + 'radii' + suffix + '.p','rb'))
    conn = pickle.load(open(path + 'connectivity' + suffix + '.p','rb'))
    xyz = pickle.load(open(path + 'verticies' + suffix + '.p','rb'))
    
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
    
    print("test")
    
    vtk_radii = vtk.vtkDoubleArray()
    for i in radii:
    	vtk_radii.InsertNextValue(i)
    vtk_radii.SetName('Radius')
    cl.GetPointData().AddArray(vtk_radii)
    
    vtk_id = vtk.vtkDoubleArray()
    for i in range(0,cl.GetNumberOfPoints()):
    	vtk_id.InsertNextValue(i)
    vtk_id.SetName('Id')
    cl.GetPointData().AddArray(vtk_id)
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(graphToPolyData.GetOutput())
    #writer.SetFileName('graph_raw_new.vtp')
    #writer.SetFileName('graph_raw_pruned.vtp')
    writer.SetFileName(path + 'graph' + suffix + '.vtp')
    writer.Write()
    
    print(cl.GetPoint(0))
    
if __name__ == '__main__':
    # executed as script
    main('/home/jszafron/Documents/source/PAMorphometry/NCH_Lungs/FN24/FN24_2/', '_reduced')