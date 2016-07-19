# to load nifti on oscar:
# module load anaconda/2-2.4.0

import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib
import pandas as pd
from skimage.measure import moments
import pickle
from skimage.measure import points_in_poly
from scipy.misc import imresize
from skimage.util.shape import view_as_windows
from skimage import measure
from skimage.filters import gaussian_filter
from skimage.measure import grid_points_in_poly
from skimage.measure import find_contours
from skimage.filters import gaussian_filter
from nifti import *
import nifti.clib as ncl
from scipy.ndimage.interpolation import zoom
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
from scipy.stats import powerlaw as sp_power


## open image files and set parameters

# Erika's top 11:
# 'image_stacks/T2'
# 'image_stacks/T3'
# 'image_stacks/Viburnum_batch12_sampleT6_redo_cropped'
# 'image_stacks/sample41_batch11_40_41_DE_stack_ok'
# 'image_stacks/sample36_batch8_FGH_stack_ok'
# 'image_stacks/sample23_batch4_EF_stack_ok'
# 'image_stacks/sample17_batch3_IJ_stack_ok'
# 'image_stacks/sample09_batch2_stack_ok'
# 'image_stacks/sample21_batch4_CD_stack_ok'
# 'image_stacks/sample10_batch2_stack_ok'
# 'image_stacks/sample06_batch2_stack_ok'

d = 'sample06_batch2_stack_ok'

files = []
for file in glob.glob('image_stacks/'+d+'/*.ti*'):
    files.append(file)

files.sort()

# open first image to get the image dimensions
im = np.array(Image.open(files[0]))

# define image_stack array
image_stack = np.zeros([len(files),np.shape(im)[0],np.shape(im)[1]])
# read and standardize all images
for i in range(len(files)):
    im = np.array(Image.open(files[i]))
    
    image_stack[i,:,:] = im

# size of the tiles
size = 60

# number of simulations to run with the SVM
n_sim = 3

# variance retained for the PCA
var = 0.90

# the contour of every step_size-th frame will be searched
step_size = 10

# number of frames to click through
nr_frames = 20


## get the training data

file_name = 'data_files/'+d+'_shell_leaf_boundaries_d1.dat'
f = open(file_name,'r')
[background_shell,shell_leaf,fs] = pickle.load(f)
f.close()

frames = np.linspace(0,len(files),num=nr_frames,endpoint=False).astype(int)

tiles = np.empty(shape=[size,size,0])
labels = np.empty(shape=[0])

for f in range(len(frames)):
    
    # calculate leaf and shell areas
    coord = shell_leaf[f]
    if len(coord) > 0:
        x = [c[0] for c in coord]
        y = [c[1] for c in coord]
        x_leaf = np.append(x,x[0])
        y_leaf = np.append(y,y[0])
        area_leaf = np.abs(np.sum(x_leaf[:-1]*y_leaf[1:]-y_leaf[:-1]*x_leaf[1:]))/2e0
        
        # expand the boundary by tile size / 3
        center_x = np.mean(x_leaf)
        center_y = np.mean(y_leaf)
        length = np.sqrt((x_leaf-center_x)**2e0 + (y_leaf-center_y)**2e0)
        x_leaf_plus = x_leaf + (x_leaf - center_x)/length*size/2e0
        y_leaf_plus = y_leaf + (y_leaf - center_y)/length*size/2e0
        # shrink the boundary by tile size / 3
        x_leaf_minus = x_leaf - (x_leaf - center_x)/length*size/2e0
        y_leaf_minus = y_leaf - (y_leaf - center_y)/length*size/2e0
    
    else:
        x_leaf = 0
        y_leaf = 0
        x_leaf_plus = 0
        y_leaf_plus = 0
        x_leaf_minus = 0
        y_leaf_minus = 0
        area_leaf = 0e0
    
    coord = background_shell[f]
    if len(coord) > 0:
        x = [c[0] for c in coord]
        y = [c[1] for c in coord]
        x_shell = np.append(x,x[0])
        y_shell = np.append(y,y[0])
        # expand the boundary by the size of the tile
        center_x = np.mean(x_shell)
        center_y = np.mean(y_shell)
        length = np.sqrt((x_shell-center_x)**2e0 + (y_shell-center_y)**2e0)
        x_shell_plus = x_shell + (x_shell - center_x)/length*size*2e0/3e0
        y_shell_plus = y_shell + (y_shell - center_y)/length*size*2e0/3e0
        
        area_shell = np.abs(np.sum(x_shell[:-1]*y_shell[1:]-y_shell[:-1]*x_shell[1:]))/2e0 - area_leaf
    
    else:
        x_shell = 0
        y_shell = 0
        x_shell_plus = 0
        y_shell_plus = 0
        area_shell = 0e0
    
    # estimate number of tiles based on the area
    
    if var == 0.99:
        n_leaf = int(area_leaf/300)
        n_shell = int(area_shell/600)
        n_background = int(n_shell/2)
    if var == 0.95:
        n_leaf = int(area_leaf/150)
        n_shell = int(area_shell/300)
        n_background = int(n_shell/1)
    if var == 0.90:
        n_leaf = int(area_leaf/60)
        n_shell = int(area_shell/180)
        n_background = int(n_shell*25)
    
    print f,n_leaf,n_shell,n_background
    
    leaf = np.zeros([size,size,n_leaf])
    shell = np.zeros([size,size,n_shell])
    background = np.zeros([size,size,n_background])
    
    i = 0
    j = 0
    k = 0
    
    while i < n_leaf or j < n_shell or k < n_background:
        
        # get random pixel from the image
        x = np.random.randint(np.shape(image_stack[frames[f],:,:])[0]-size)
        y = np.random.randint(np.shape(image_stack[frames[f],:,:])[1]-size)
        
        # check whether all the corners of the square fall within any of the regions
        
        corners = np.zeros([4,2])
        corners[0,:] = [x,y]
        corners[1,:] = [x+size,y]
        corners[2,:] = [x,y+size]
        corners[3,:] = [x+size,y+size]
        
        
        if np.all(points_in_poly(corners,np.column_stack((x_leaf_plus,y_leaf_plus)))) and i < n_leaf:
            # leaf
            # get the hog features of this cell
            leaf[:,:,i] = image_stack[frames[f],x:x+size,y:y+size]
            square = plt.Rectangle((x,y),size,size,color='r',fill=False)
            plt.gca().add_patch(square)
            i = i + 1
        
        
        if np.all(points_in_poly(corners,np.column_stack((x_shell_plus,y_shell_plus)))) and np.all(~points_in_poly(corners,np.column_stack((x_leaf_minus,y_leaf_minus)))) and j < n_shell:
            # shell
            shell[:,:,j] = image_stack[frames[f],x:x+size,y:y+size]
            square = plt.Rectangle((x,y),size,size,color='y',fill=False)
            plt.gca().add_patch(square)
            j = j + 1
        
        
        if np.all(~points_in_poly(corners,np.column_stack((x_shell_plus,y_shell_plus)))) and k < n_background:
            # background
            background[:,:,k] = image_stack[frames[f],x:x+size,y:y+size]
            square = plt.Rectangle((x,y),size,size,color='b',fill=False)
            plt.gca().add_patch(square)
            k = k + 1

tiles = np.append(tiles,np.concatenate((leaf,shell,background),axis=2),axis=2)
    label = np.concatenate((np.zeros(n_leaf)+2,np.zeros(n_shell)+1,np.zeros(n_background)))
    labels = np.append(labels,label)
    
    plt.axis('equal')
    
    plt.xlim([0,np.shape(image_stack)[1]])
    plt.ylim([0,np.shape(image_stack)[2]])
    plt.imshow(image_stack[frames[f],:,:].T,cmap='Greys_r')
    plt.savefig('imgs/shell_leaf_boundaries_'+d+'_'+str(frames[f])+'_size'+str(size)+'.png',dpi=150)
    plt.close()


print np.shape(tiles)
print np.shape(labels)
print len(labels[labels == 2]),len(labels[labels == 1]),len(labels[labels == 0])

file_name = 'data_files/training_data_'+d+'_'+str(size)+'x'+str(size)+'_var0'+str(int(var*100e0))+'.dat'
f = open(file_name,'w')
pickle.dump([tiles,labels],f)
f.close()



## do PCA
'''
    Performs the Principal Coponent analysis of the Matrix X
    Matrix must be n * m dimensions
    where n is # features
    m is # examples
    '''

def PCA(X, varRetained = [0.95],filename = 'PCA_data.dat'):
    
    # Compute Covariance Matrix Sigma
    (n, m) = X.shape
    
    Sigma = 1.0 / float(m) * np.dot(X, np.transpose(X))
    # Compute eigenvectors and eigenvalues of Sigma
    U, s, V = np.linalg.svd(Sigma)
    
    # compute the value k: number of minumum features that
    # retains the given variance
    s_tot = np.sum(s)
    
    var_i = np.array([np.sum(s[: i + 1]) / s_tot * 100.0 for i in range(n)])
    
    k = np.zeros(len(varRetained))
    for i in range(len(k)):
        k[i] = len(var_i[var_i < (varRetained[i] * 100e0)])
        
        print '%.2f %% variance retained in %d dimensions' % (var_i[k[i]], k[i])
        
        # compute the reduced dimensional features
        U_reduced = U[:, : k[i]]
        Z = np.dot(np.transpose(U_reduced),X)
        
        # pickle dump the results
        f = open(filename+str(int(varRetained[i]*100e0))+'.dat','w')
        pickle.dump([Z, U_reduced, k[i]],f)
        f.close()
    
    return


var_ret = [0.90, 0.95, 0.99]

# load the training data and divide it to training (60%), test (20%), and cross validation (20%) sets
f = open('data_files/training_data_'+d+'_'+str(size)+'x'+str(size)+'_var0'+str(int(var*100e0))+'.dat','r')
[tiles,labels] = pickle.load(f)
f.close()


# standardize tiles
tiles_standard = np.zeros(np.shape(tiles))

mean = np.mean(image_stack)
std = np.std(image_stack)
for i in range(len(labels)):
    tiles_standard[:,:,i] = (tiles[:,:,i] - mean) / std

# reshape tiles
reshape_tiles = np.reshape(tiles_standard,[size*size,len(labels)])

filename = 'data_files/PCA_data_'+d+'_'+str(size)+'x'+str(size)+'_var0'

# do PCA and save the results
PCA(reshape_tiles,varRetained = var_ret,filename = filename)

print 'finished'

# In[2]:

# train a random forest with RandomizedSearchCV

file_name = 'data_files/training_data_'+d+'_'+str(size)+'x'+str(size)+'_var0'+str(int(var*100e0))+'.dat'
f = open(file_name,'r')
[X,Y] = pickle.load(f)
f.close()

f = open('data_files/PCA_data_'+d+'_'+str(size)+'x'+str(size)+'_var0'+str(int(var*100e0))+'.dat','r')
[Z, U_reduced, k] = pickle.load(f)
f.close()

# initialize the classifier
RF = RandomForestClassifier()

# parameter grid for randomized search
param_grid = {"max_depth": sp_randint(1, 20),
    "max_features": sp_randint(1, np.shape(Z.T)[1]),
        "min_samples_leaf": sp_randint(1, 100),
            "class_weight": ["auto"],
                "n_jobs": [16],
                    "n_estimators": sp_randint(10, 100)}

cv = StratifiedKFold(Y,n_folds=k_fold,shuffle=True)

# do the parameter search
search_RF = RandomizedSearchCV(RF,param_grid,n_iter=n_iter,cv=cv).fit(Z.T,Y)
print '   ',search_RF.best_score_
print '   ',search_RF.best_params_

# save the results
f = open('data_files/RF_tile_'+d+'_niter'+str(n_iter)+'_kfold'+str(k_fold)+'_var0'+str(int(var*100e0))+'.dat','w')
pickle.dump([search_RF.grid_scores_,search_RF.best_score_,search_RF.best_params_,search_RF.best_estimator_],f)
f.close()



# In[3]:

# train an XGBoost with RandomizedSearchCV

# initialize the classifier
GB = xgb.XGBClassifier()

# parameter grid for randomized search
param_grid = {"max_depth": sp_randint(1, 20),
    "learning_rate": sp_uniform(loc=0e0,scale=1e0),
        "nthread":[16],
            "objective":['multi:softprob'],
                "n_estimators": sp_randint(50, 200)}

cv = StratifiedKFold(Y,n_folds=k_fold,shuffle=True)

# do the parameter search
search_GB = RandomizedSearchCV(GB,param_grid,n_iter=n_iter,cv=cv).fit(Z.T,Y)
print '   ',search_GB.best_score_
print '   ',search_GB.best_params_

# save the results
f = open('data_files/GB_tile_'+d+'_niter'+str(n_iter)+'_kfold'+str(k_fold)+'_var0'+str(int(var*100e0))+'.dat','w')
pickle.dump([search_GB.grid_scores_,search_GB.best_score_,search_GB.best_params_,search_GB.best_estimator_],f)
f.close()



# In[6]:

# train an rbf SVM with GridSearchCV

# initialize the classifier
SVM_rbf = SVC()

# parameter grid for randomized search
param_grid = {"C": 10e0**(np.linspace(0e0,4e0,5)),
    "class_weight": ["auto"],
        "gamma": 10e0**(np.linspace(-4e0,0e0,5))}

cv = StratifiedKFold(Y,n_folds=k_fold,shuffle=True)

# do the parameter search
search_SVM = GridSearchCV(SVM_rbf,param_grid,cv=cv,n_jobs=8).fit(Z.T,Y)
print '   ',search_SVM.best_score_
print '   ',search_SVM.best_params_

# save the results
f = open('data_files/SVM_rbf_'+d+'_tile_niter'+str(n_iter)+'_kfold'+str(k_fold)+'_var0'+str(int(var*100e0))+'.dat','w')
pickle.dump([search_SVM.grid_scores_,search_SVM.best_score_,search_SVM.best_params_,search_SVM.best_estimator_],f)
f.close()



# In[3]:

#render the classes of all classifiers

f = open('data_files/PCA_data_'+d+'_'+str(size)+'x'+str(size)+'_var0'+str(int(var*100e0))+'.dat','r')
[Z, U_reduced, k] = pickle.load(f)
f.close()

f = open('data_files/RF_tile_'+d+'_niter'+str(n_iter)+'_kfold'+str(k_fold)+'_var0'+str(int(var*100e0))+'.dat','r')
[RF_grid_scores,RF_best_score,RF_best_params,RF] = pickle.load(f)
f.close()

f = open('data_files/GB_tile_'+d+'_niter'+str(n_iter)+'_kfold'+str(k_fold)+'_var0'+str(int(var*100e0))+'.dat','r')
[GB_grid_scores,GB_best_score,GB_best_params,GB] = pickle.load(f)
f.close()

f = open('data_files/SVM_rbf_'+d+'_tile_niter'+str(n_iter)+'_kfold'+str(k_fold)+'_var0'+str(int(var*100e0))+'.dat','r')
[SVM_rbf_grid_scores,SVM_rbf_best_score,SVM_rbf_best_params,SVM_rbf] = pickle.load(f)
f.close()


windows = view_as_windows(image_stack[0,:,:],(size,size))

f_range = np.arange(np.shape(image_stack)[0],step=step_size)
i_range = np.arange(np.shape(windows)[0],step=step_size)
j_range = np.arange(np.shape(windows)[1],step=step_size)

class_stack_RF = np.zeros([len(f_range),len(i_range),len(j_range)]).astype(int)
class_stack_SVM_rbf = np.zeros([len(f_range),len(i_range),len(j_range)]).astype(int)
class_stack_GB = np.zeros([len(f_range),len(i_range),len(j_range)]).astype(int)


mean_im = np.mean(image_stack)
std_im = np.std(image_stack)
for f in range(len(f_range)):
    if f%(100/step_size) == 0:
        print f
    
    windows = view_as_windows(image_stack[f_range[f],:,:],(size,size))
    
    PCA_features = np.zeros([len(i_range),len(j_range),np.shape(Z)[0]])
    # collect the PCA features
    for i in range(len(i_range)):
        for j in range(len(j_range)):
            tile_standard = (windows[i_range[i],j_range[j]] - mean_im)/std_im
            reshape_tile = tile_standard.reshape(size*size)
            PCA_features[i,j,:] = np.dot(reshape_tile,U_reduced)

# predict the classes:
for i in range(len(i_range)):
    class_stack_RF[f,i,:] = RF.predict(PCA_features[i,:,:])
        class_stack_SVM_rbf[f,i,:] = SVM_rbf.predict(PCA_features[i,:,:])
        class_stack_GB[f,i,:] = GB.predict(PCA_features[i,:,:])


f = open('data_files/class_stacks_'+d+'_'+str(size)+'x'+str(size)+'_var0'+str(int(var*100e0))+'.dat','w')
pickle.dump([class_stack_RF,class_stack_SVM_rbf,class_stack_GB],f)
f.close()



# In[13]:

f = open('data_files/class_stacks_'+d+'_'+str(size)+'x'+str(size)+'_var0'+str(int(var*100e0))+'.dat','r')
#pickle.dump([class_stack_RF,class_stack_SVM_rbf,class_stack_SVM_lin,class_stack_SVM_poly,class_stack_GB,class_stack_knearest],f)
[class_stack_RF,class_stack_SVM_rbf,class_stack_GB] = pickle.load(f)
f.close()

class_stack_RF[class_stack_RF == 1] = 0
class_stack_SVM_rbf[class_stack_SVM_rbf == 1] = 0
class_stack_GB[class_stack_GB == 1] = 0

# combine the class stacks
class_stack = class_stack_RF + class_stack_SVM_rbf + class_stack_GB

#class_stack = class_stack_RF + class_stack_SVM_rbf
#class_stack = class_stack_RF + class_stack_GB
#class_stack = class_stack_SVM_rbf + class_stack_GB


class_stack[class_stack < 4] = 0
class_stack[class_stack >= 4] = 1

boundary = np.zeros(np.shape(image_stack))

for f in range(np.shape(image_stack)[0]):
    
    if f%50 == 0:
        print f
    
    smoothed = gaussian_filter(class_stack[f/step_size,:,:].astype(float),sigma=2)
    
    contours = find_contours(smoothed, 0.5)
    
    if len(contours) != 0:
        
        # calculate the area of the contours
        area = np.zeros(len(contours))
        
        for n, contour in enumerate(contours):
            x = contour[:,0]
            y = contour[:,1]
            # copy the first point to the end of the contour list to close the loop
            x = np.append(x,x[0])
            y = np.append(y,y[0])
            # use Green's theorem to calculate the area of the contour
            area[n] = np.abs(np.sum(x[:-1]*y[1:]-y[:-1]*x[1:]))/2e0
        
        # find contour with the largest area
        cont = contours[np.where(area == np.max(area))[0]]
    
    pixels_in_cont = grid_points_in_poly(np.shape(image_stack[f,:,:]),cont*step_size+size/2e0)
    
    boundary[f,:,:] = image_stack[f,:,:]*pixels_in_cont


nim = NiftiImage(zoom(boundary,0.5))
print nim.header['dim']
nim.header['datatype'] == ncl.NIFTI_TYPE_FLOAT64
nim.save('data_files/'+d+'_boundary_half_res_size'+str(size)+'.nii.gz')

print 'done'