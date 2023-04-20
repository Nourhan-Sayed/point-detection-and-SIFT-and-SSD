import cv2 as cv
import numpy as np
from scipy import signal
from scipy import linalg
import time
import matplotlib.pyplot as plt


def lambdamin(img,window=3,q=0.999):
    
    # first convert to grayscale if RGB
    if len(img)>=3:
        img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    # smooth the image 
    img= cv.GaussianBlur(img,(5,5),0)
        
    
    # find gradients by sobel derivative 
    
    sobelx = np.array([[ -1 , 0 , 1 ] ,
                    [ -2 , 0 , 2 ] ,
                    [ -1 , 0 , 1 ]])
    sobely = sobelx.transpose()
    
    # spatial derivatives
    
    ix=signal.convolve2d( img , sobelx ,'same') 
    iy=signal.convolve2d( img , sobely ,'same') 
    
    # calculate structure tensor elements 
    ixx=np.multiply( ix, ix) # squared ix
    iyy=np.multiply( iy, iy) # squared iy 
    ixy=np.multiply( ix, iy) # ix . iy
    
    # improve performance by applying a window
    
    ixx=cv.blur(ixx,(window,window))
    iyy=cv.blur(iyy,(window,window))
    ixy=cv.blur(ixy,(window,window))
    
    # get H matrix for each element and find lambda min
    rows,cols= img.shape
    lambdamat=np.zeros_like(img)
    for i in range (rows):
        for j in range (cols):
            H=[[ixx[i,j],ixy[i,j]],[ixy[i,j],iyy[i,j]]]
            #print(H)
            eigvals=linalg.eigvals(H)
            try: 
                lambdamin= np.min(eigvals[np.nonzero(eigvals)]) # to avoid getting 0 as eigen value for all pixels
            except :
                lambdamin=0  # when both are zero we set it to zero
                
                
            lambdamat[i,j]=lambdamin
            
    # apply non maximal supression to get the highest lambda values
    
    lambdamat=np.abs(lambdamat) >  np.quantile( np.abs(lambdamat),q)

            
    return lambdamat
    
img=cv.imread("chess.png")   
corners=lambdamin(img,window=2,q=0.995)
plt.imshow(img,zorder=1)
corners = np.argwhere(corners)

plt.scatter(corners[:,1],corners[:,0],zorder=2, c = 'r',marker ='o')
plt.show()