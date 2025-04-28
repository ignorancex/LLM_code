import nilearn.image as ni
import numpy as np
from scipy.ndimage import binary_fill_holes
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from skimage import measure
from surfproc import smooth_patch
from dfsio import writedfs
import vtk.util.numpy_support as numpy_support

import vtk
import sys

from scipy.ndimage.morphology import grey_dilation

def MarchingCubes(image,threshold):
    '''
    http://www.vtk.org/Wiki/VTK/Examples/Cxx/Modelling/ExtractLargestIsosurface 
    '''
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.ComputeGradientsOn()
    mc.SetValue(0, threshold)
    mc.Update()
    
    # To remain largest region
    confilter =vtk.vtkPolyDataConnectivityFilter()
    confilter.SetInputData(mc.GetOutput())
    confilter.SetExtractionModeToLargestRegion()
    confilter.Update()

    return confilter.GetOutput()




    
label_file = '/deneb_disk/feta_2022/test/sub-058/anat/sub-058_rec-irtk_T2w66.label.nii.gz'
label_dilated_file = '/deneb_disk/feta_2022/test/sub-058/anat/sub-058_rec-irtk_T2w66.dilated.label.nii.gz'

gmpial_file = 'gmpial.nii.gz'
wmin_file = 'wmin.nii.gz'

pial_file = '/deneb_disk/feta_2022/test/sub-058/anat/pial.dfs'
inner_file = '/deneb_disk/feta_2022/test/sub-058/anat/inner.dfs'

# Convert to int16
#v = ni.load_img(label_file)
#v2=ni.new_img_like(label_file,np.int16(v.get_fdata()))
#v2.to_filename(label_file2)

# write dilated label file
img2=v.get_fdata()
img2[img2 == 1]=0
img2=grey_dilation(img2,[5,5,5])

label_dilated = ni.new_img_like(label_file,np.int16(img2))
label_dilated.to_filename(label_dilated_file)

# Create CSF image
img = np.int16((v.get_fdata() != 124) & (v.get_fdata() > 0))
csf = ni.new_img_like(label_file, img)
csf = ni.reorder_img(csf)
csf.to_filename(gmpial_file)

# Create WM file
v=ni.load_img(tissue_file)
WM_in = np.int16((v.get_fdata() != 112) & (v.get_fdata() != 113) & (v.get_fdata() != 124) & (v.get_fdata() > 0)) 
wm= ni.new_img_like(label_file,WM_in)
wm = ni.reorder_img(wm)
wm.to_filename(wmin_file)

ff = wmin_file # THIS FILE DETERMINES WHAT SURFACE IS GENERATED
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(ff)
reader.Update()
im = reader.GetOutput()
poly = MarchingCubes(im,.5)
#visualize_poly(poly)
writer = vtk.vtkSTLWriter()
writer.SetInputData(poly)
writer.SetFileName('xx.stl')
writer.Update()

pts = poly.GetPoints()
vertices = vtk_to_numpy(pts.GetData())
faces = poly.GetPolys()
faces = vtk_to_numpy(faces.GetData())
faces = faces.reshape(np.int32(len(faces) / 4), 4)
faces = faces[:, (1,3,2)]
print(vertices.shape, faces.shape)

class s:
    pass

s.faces = faces
s.vertices = vertices

s = smooth_patch(s,10)

writedfs(inner_file,s)