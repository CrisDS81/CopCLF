# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:44:21 2021

@author: Cristiano
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:15:21 2021

@author: Cristiano
"""
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




def duplodata_all_preprocess():
    filepath_B2_all = r'E:Reunion_Island_INRAE/S2_THEIA_40KCB_B2_GAPF.tif'
    filepath_B3_all = r'E:Reunion_Island_INRAE/S2_THEIA_40KCB_B3_GAPF.tif'
    filepath_B4_all = r'E:Reunion_Island_INRAE/S2_THEIA_40KCB_B4_GAPF.tif'
    filepath_B8_all = r'E:Reunion_Island_INRAE/S2_THEIA_40KCB_B8_GAPF.tif'
    filepath_NDVI_all = r'E:Reunion_Island_INRAE/S2_THEIA_40KCB_NDVI_GAPF.tif'
    #filepath_gt_all = r'E:Reunion_Island_INRAE/BD_Reunion_V2_CodeN3.tif'
    filepath_gt_all = r'E:Reunion_Island_INRAE/BD_Reunion_V2_ObjID.tif'
    #filepath_gt_all = r'E:Reunion_Island_INRAE/groundtruth_class_pixel.tif'
    
    raster = gdal.Open(filepath_gt_all)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    gt_all=fullarray.reshape(-1,1)
    n=raster.RasterXSize
    m=raster.RasterYSize
    gt_all=gt_all.reshape(n*m,1)
    rows=np.where(gt_all!=0)[0]
    gt_all=gt_all[rows,:]
    
    raster = gdal.Open(filepath_gt_all)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    gt_all_obj=fullarray.reshape(-1,1)
    n=raster.RasterXSize
    m=raster.RasterYSize
    gt_all=gt_all_obj.reshape(n*m,1)
    rows=np.where(gt_all_obj!=0)[0]
    gt_all_obj=gt_all_obj[rows,:]
    
    
    raster = gdal.Open(filepath_B2_all)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B2_all=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B2_all=B2_all.reshape(n*m,34)
    B2_all=B2_all[rows,:]
    B2_all=np.where(B2_all==-9999,0,B2_all)
    
    temp_row=np.where(B2_all[:,0]!=-9999)[0]
    B2_all=B2_all[temp_row,:]
    n_comp=select_n_components(B2_all, 0.95)
    B2_all_red=dim_reduction(B2_all, n_components=n_comp)
    
    
    raster = gdal.Open(filepath_B3_all)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B3_all=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B3_all=B3_all.reshape(n*m,34)
    B3_all=B3_all[rows,:]
    B3_all=np.where(B3_all==-9999,0,B3_all)

    temp_row=np.where(B3_all[:,0]!=-9999)[0]
    B3_all=B3_all[temp_row,:]
    n_comp=select_n_components(B3_all, 0.95)
    B3_all_red=dim_reduction(B3_all, n_components=n_comp)
    
    
    raster = gdal.Open(filepath_B4_all)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B4_all=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B4_all=B4_all.reshape(n*m,34)
    B4_all=B4_all[rows,:]
    B4_all=np.where(B4_all==-9999,0,B4_all)
    
    temp_row=np.where(B4_all[:,0]!=-9999)[0]
    B4_all=B4_all[temp_row,:]
    n_comp=select_n_components(B4_all, 0.95)
    B4_all_red=dim_reduction(B4_all, n_components=n_comp)
    
    
    raster = gdal.Open(filepath_B8_all)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    B8_all=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    B8_all=B8_all.reshape(n*m,34)
    B8_all=B8_all[rows,:]
    B8_all=np.where(B8_all==-9999,0,B8_all)
    
    temp_row=np.where(B8_all[:,0]!=-9999)[0]
    B8_all=B8_all[temp_row,:]
    n_comp=select_n_components(B8_all, 0.95)
    B8_all_red=dim_reduction(B8_all, n_components=n_comp)
    
    
    raster = gdal.Open(filepath_NDVI_all)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    ndvi_all=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    ndvi_all=ndvi_all.reshape(n*m,34)
    ndvi_all=ndvi_all[rows,:]
    ndvi_all=np.where(ndvi_all==-9999,0,ndvi_all)
    
    temp_row=np.where(ndvi_all[:,0]!=-9999)[0]
    ndvi_all=ndvi_all[temp_row,:]
    n_comp=select_n_components(ndvi_all, 0.95)
    ndvi_all_red=dim_reduction(ndvi_all, n_components=n_comp)
    
    # svd = TruncatedSVD(n_components=n_comp)#7 is the best for now
    # # prepare transform on dataset
    # svd.fit(ndvi_all)
    # # apply transform to dataset
    # transformed = svd.transform(ndvi_all)
    # ndvi_all_red=transformed.copy()
    
    gt_all=gt_all[temp_row]
    
    X_label=np.concatenate((B2_all,B3_all,B4_all,B8_all,ndvi_all,gt_all),axis=1)
    X_label_red=np.concatenate((B2_all_red,B3_all_red,B4_all_red,B8_all_red,ndvi_all_red,gt_all),axis=1)
    #X_no_ndvi_label=np.concatenate((B2_all,B3_all,B4_all,B8_all,gt_all),axis=1)
    
    #X_ndvi_label=np.concatenate((ndvi_all,gt_all),axis=1)
    
    X_no_label=X_label[:,0:170]
    X_no_label_red=X_label_red[:,0:19]
        
    #X_no_ndvi_no_label=X_no_ndvi_label[:,0:170]
    
    #X_ndvi_no_label=X_ndvi_label[:,0:170]
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_label_red_reunion_island.csv",X_label_red, delimiter=",")
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_no_label_red_reunion_island.csv",X_no_label_red, delimiter=",")
    
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_label_reunion_island.csv",X_label, delimiter=",")
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_no_label_reunion_island.csv",X_no_label, delimiter=",")
    
    np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/gt_all_obj.csv",gt_all_obj, delimiter=",") 
    #np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_no_ndvi_label_reunion_island.csv",X_no_ndvi_label, delimiter=",")
    
    # np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_no_ndvi_no_label_reunion_island.csv",X_no_ndvi_no_label, delimiter=",")
    
    # np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_ndvi_label_reunion_island.csv",X_ndvi_label, delimiter=",")

    # np.savetxt("E:/Reunion_Island_INRAE/Reunion_island_all/X_ndvi_no_label_reunion_island.csv",X_ndvi_no_label, delimiter=",")

    X_train, X_test, y_train, y_test = train_test_split(X_no_label, gt_all, stratify=gt_all, test_size=0.3)
    return X_train, X_test, y_train, y_test

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

def senegaldata_exp():
    #scegliere 44 o 45 o 54 o 55 che sono le divisioni centrali
    filepath = r'E:Senegal_Inrae/Senegal_INRAE/Senegal_exp/Senegal_NDVI_path_3.tif'
    filepath_gt = r'E:Senegal_Inrae/Senegal_INRAE/Senegal_exp/ground-truth/Senegal_NDVI_path_3_gt.tif'

    raster = gdal.Open(filepath)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    chunk_1=np.moveaxis(fullarray,0,2)
    n=raster.RasterXSize
    m=raster.RasterYSize
    d=raster.RasterCount
    chunk_1=chunk_1.reshape(n*m,d)
    
    raster = gdal.Open(filepath_gt)
    raster.GetProjection()
    fullarray = np.array(raster.ReadAsArray())
    gt=fullarray
    n=raster.RasterXSize
    m=raster.RasterYSize
    gt=gt.reshape(n*m,)
    gt=np.where(gt==0,gt,1)
    X=chunk_1.copy()
    X_label=np.concatenate((X,gt.reshape(-1,1)), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, gt, stratify=gt, test_size=0.3)
    return [X_train, X_test, y_train, y_test,X, X_label, gt]