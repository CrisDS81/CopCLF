

import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
from numpy import linalg as LA

pre = rasterio.open('C:/Users/Cristiano/Desktop/Python_Code/Burned_Area_Dat/sentinel2_MS_SubArea_pre.tif')
post= rasterio.open('C:/Users/Cristiano/Desktop/Python_Code/Burned_Area_Dat/sentinel2_MS_SubArea_post.tif')
array = pre.read(1)
array_post= post.read(1)
plt.imshow(array_post, cmap='pink')
gt = rasterio.open('C:/Users/Cristiano/Desktop/Python_Code/Burned_Area_Dat/gt.tif')
pre_ = pre.read()
post_ = post.read()
gt_ = gt.read()
img_pre = reshape_as_image(pre_)
img_post = reshape_as_image(post_)
img_gt = reshape_as_image(gt_)
img_pre_matrix=img_pre.reshape(img_pre.shape[0]*img_pre.shape[1], img_pre.shape[2])
img_post_matrix=img_post.reshape(img_post.shape[0]*img_post.shape[1], img_post.shape[2])
img_gt_matrix=img_gt.reshape(img_gt.shape[0]*img_gt.shape[1], img_gt.shape[2])



svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42)
svd.fit(img_pre_matrix)
new_img_pre=svd.transform(img_pre_matrix)
svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42)
svd.fit(img_post_matrix)
new_img_post=svd.transform(img_post_matrix)


diff_nbr= rasterio.open('C:/Users/Cristiano/Desktop/Python_Code/Burned_Area_Dat/diff_nbr.tif')
diff_nbr_ = diff_nbr.read()
img_diff_nbr = reshape_as_image(diff_nbr_)
plt.imshow(img_diff_nbr, cmap=plt.cm.gray)


img_diff_nbr_matrix=img_diff_nbr.reshape(img_diff_nbr.shape[0]*img_diff_nbr.shape[1], img_diff_nbr.shape[2])

def euclideanDistance(A, B):
    if(A.shape==B.shape):
        print("------Euclidean distance")
        row=A.shape[0]
        euc=np.zeros([row,1], float)
        diff=A-B
        for i in range(row):
            euc[i]=LA.norm(diff[i].T, 2)        
        return  euc
    else:
        print("Can't calculate euclidean distance, matrixs have not the same shape")
        print("A: "+str(A.shape))
        print("B: "+str(B.shape))
        
        
def SAM(A, B):   
    if(A.shape==B.shape):
        print("------SAM")
        row, k = A.shape
        sam = np.zeros([row,1], float)
        for i in range(row):
            normT1=np.linalg.norm(A[i, :])
            normT2=np.linalg.norm(B[i, :])
  
            product=np.dot(A[i, :], B[i, :])
            sam[i] = np.arccos( product /max((normT2 * normT1), 1e-5))
            # to add 2/pi
        return sam
    else:
        print("Can't calculate SAM, matrix dimensions are not equal:")
        print("A = "+str(A.shape))
        print("B = "+str(B.shape))
        
        
def SAM_MEAN(A, B, windowSize=2):
    if(A.shape==B.shape):
        print("------SAM mean")
        row, col, feature = A.shape
        C= np.zeros([row, col,1], float)
        
        for i in range(row):
            for j in range (col):
               
                iS=max(i-windowSize, 0)
                iE=min(i+windowSize+1, row)
                jS=max(j-windowSize, 0)
                jE=min(j+windowSize+1, col)
              
                windowArea=((iE-iS)*(jE-jS))
                #AwindowMean=sum(sum(sum((A[iS:iE, jS:jE])[:,:])[:]))/((feature)*(windowArea))      
                #BwindowMean=sum(sum(sum((B[iS:iE, jS:jE])[:,:])[:]))/((feature)*(windowArea)) 
                den1 = np.sqrt(np.sum(np.multiply(A[iS:iE, jS:jE], A[iS:iE, jS:jE]), axis=2))
                den1[den1 < 1e-5]=1e-5
                den2 = np.sqrt(np.sum(np.multiply(B[iS:iE, jS:jE], B[iS:iE, jS:jE]), axis=2))
                den2[den2 < 1e-5]=1e-5
                
                
          
                den = den1*den2
                # angolo della media dei coseni
                #num = np.arccos(np.sum(np.divide(np.sum(np.multiply(A[iS:iE, jS:jE],B[iS:iE, jS:jE]),axis=2),den))/windowArea)
                #angolo medio
                num = np.sum(np.arccos(np.divide(np.sum(np.multiply(A[iS:iE, jS:jE],B[iS:iE, jS:jE]),axis=2),den)))
             
                #den1=sum(sum(sum(np.multiply(A[iS:iE, jS:jE]-AwindowMean, A[iS:iE, jS:jE]-AwindowMean)[:,:])[:]))
                #den2=sum(sum(sum(np.multiply(B[iS:iE, jS:jE]-BwindowMean, B[iS:iE, jS:jE]-BwindowMean)[:,:])[:]))
                
                calcolo=num/windowArea
                #calcolo=num
                C[i,j]=calcolo
        return C
    else:
        print("Can't calculate Correlation, matrix dimensions are not equal:")
        print("A = "+str(A.shape))
        print("B = "+str(B.shape))
        return
    

SAM_MEAN_dist=SAM_MEAN(new_img_pre.reshape(1866,2019,3), new_img_post.reshape(1866,2019,3), windowSize=2)
SAM_dist=SAM(new_img_pre,new_img_post)
def nbr(band1, band2):
    """
    This function takes an input the arrays of the bands from the read_band_image
    function and returns the Normalized Burn ratio (NBR)
    input:  band1   array (n x m)      array of first band image e.g B8A
            band2   array (n x m)      array of second band image e.g. B12
    output: nbr     array (n x m)      normalized burn ratio
    """
    nbr = (band1 - band2) / (band1 + band2)
    return nbr

band8=post.read(7).reshape(-1,1)
band12=post.read(11).reshape(-1,1)
nbr_post=nbr(band8, band12)


