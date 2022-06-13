
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from osgeo import gdal
from pathlib import Path
import pandas as pd
from utils.other_utils import select_n_components, dim_reduction
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, IncrementalPCA
#from fast_ml.model_development import train_valid_test_split
import os
os.environ["PROJ_LIB"]="C:/OSGeo4W64/share/proj"

def duplodata_all(path):
    X_label=pd.read_csv(path/'X_label_reunion_island.csv')
    X_no_label=pd.read_csv(path/'X_no_label_reunion_island.csv')
    X_label=X_label.values
    gt=X_label[:,-1]
    X_no_label=X_no_label.values
    X_train, X_test, y_train, y_test = train_test_split(X_no_label, gt, stratify=gt, test_size=0.3)
    return [X_train, X_test, y_train, y_test, X_label, gt]

def duplodata_all_val(path):
    X_label=pd.read_csv(path/'X_label_reunion_island.csv')
    X_no_label=pd.read_csv(path/'X_no_label_reunion_island.csv')
    X_label=X_label.values
    gt=X_label[:,-1]
    X_no_label=X_no_label.values
    X_train, X_test, y_train, y_test= train_test_split(X_no_label, gt, stratify=gt, test_size=0.7, random_state=1)
    X_val, X_test, y_val, y_test  = train_test_split(X_test, y_test,stratify=y_test, test_size=0.5)
    return [X_train, X_test, y_train, y_test, X_label, gt]




def duplodata_obj(path):
    gt_obj=pd.read_csv(path/'gt_all_obj.csv')
    X_label=pd.read_csv(path/'X_label_reunion_island.csv')
    X_no_label=pd.read_csv(path/'X_no_label_reunion_island.csv')
    aa=np.concatenate((X_label,gt_obj), axis=1)
    bb = []
    for i in np.unique(aa[:,171]):
        tmp = aa[np.where(aa[:,171] == i)]
        bb.append(np.mean(tmp,axis=0))
    bb=np.stack(bb)
    bb=bb[:,:171]
    gt=bb[:,170]
    X_train, X_test, y_train, y_test= train_test_split(bb[:,:170], gt, stratify=gt, test_size=0.3, random_state=1)
    return [X_train, X_test, y_train, y_test, X_label, gt]
    

def duplodata_all_red(path):
    X_label=pd.read_csv(path/'X_label_red_reunion_island.csv')
    X_no_label=pd.read_csv(path/'X_no_label_red_reunion_island.csv')
    X_label=X_label.values
    gt=X_label[:,-1]
    X_no_label=X_no_label.values
    X_train, X_test, y_train, y_test = train_test_split(X_no_label, gt, stratify=gt, test_size=0.4, random_state=400)
    return [X_train, X_test, y_train, y_test, X_label, gt]  
    
    
    
def duplodata_exp():
    #scegliere 44 o 45 o 54 o 55 che sono le divisioni centrali
    filepath_B2 = r'E:Reunion_Island_INRAE/Reunion_Example_paper/Ex_5_B2.tif'
    filepath_B3 = r'E:Reunion_Island_INRAE/Reunion_Example_paper/Ex_5_B3.tif'
    filepath_B4 = r'E:Reunion_Island_INRAE/Reunion_Example_paper/Ex_5_B4.tif'
    filepath_B8 = r'E:Reunion_Island_INRAE/Reunion_Example_paper/Ex_5_B8.tif'
    filepath_NDVI = r'E:Reunion_Island_INRAE/Reunion_Example_paper/Ex_5_NDVI.tif'
    filepath_gt = r'E:Reunion_Island_INRAE/Reunion_Example_paper/Ex_5_gt.tif'
    
    raster = gdal.Open(filepath_gt)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    gt=fullarray
    n=raster.RasterXSize
    m=raster.RasterYSize
    gt=gt.reshape(n*m,)


    raster = gdal.Open(filepath_B2)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B2=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B2=B2.reshape(n*m,34)
    #B2=B2[rows,:]
    
    raster = gdal.Open(filepath_B3)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B3=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B3=B3.reshape(n*m,34)
    #B3=B3[rows,:]
    
    raster = gdal.Open(filepath_B4)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B4=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B4=B4.reshape(n*m,34)
    #B4=B4[rows,:]
    
    raster = gdal.Open(filepath_B8)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B8=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B8=B8.reshape(n*m,34)
    #B8=B8[rows,:]
    
    raster = gdal.Open(filepath_NDVI)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    ndvi=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    ndvi=ndvi.reshape(n*m,34)
    #ndvi=ndvi[rows,:]
    

    X_exp=np.concatenate((B2,B3,B4,B8,ndvi), axis=1)
    X_label_exp=np.concatenate((X_exp,gt.reshape(-1,1)),axis=1)
    
    rows=np.where(gt!=0)[0]
    gt=gt[rows]
   
    X_exp=X_exp[rows,:]
 
    '''check if labels start from 0, otherwise change'''
    '''
    if np.min(merged_labels)==1:
        merged_labels=merged_labels-1
    if np.min(labels_test)==1:
        labels_test=labels_test-1
    
    y_train=merged_labels.copy()
    y_test=labels_test.copy()
    X_test=ts_test.copy()
    '''
    X_train, X_test, y_train, y_test = train_test_split(X_exp, gt, stratify=gt, test_size=0.3)
    return X_train, X_test, y_train, y_test
