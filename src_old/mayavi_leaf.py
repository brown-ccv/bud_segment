# animate some buds and leaves

import os
import numpy as np
from nifti import *
import nifti.clib as ncl
from mayavi import mlab

# collect segmentations
segs = []
for file in os.listdir("data_files/"):
    if file.endswith(".nii.gz") and ('segmented' in file):
        segs.append(file)
        
for s in segs:
    nim = NiftiImage('data_files/'+s).asarray()

    print s,np.shape(nim)
    if 'sample17' in s:
        nim[nim ==1] = 0
        nim_plot = np.swapaxes(nim,0,2)
        for i in range(360):
            # make the plot with mlab
            fig = mlab.figure(bgcolor=(1,1,1),size=(480,640))

            mlab.contour3d(nim_plot,figure=fig)
            #mlab.triangular_mesh([vert[0] for vert in verts],
            #          [vert[1] for vert in verts],
            #          [vert[2] for vert in verts], faces) 
            mlab.view(azimuth = i,distance=1200e0)
            mlab.savefig('animation_border/leaf/leaf_'+s+'_'+str(i).zfill(3)+'.png')
