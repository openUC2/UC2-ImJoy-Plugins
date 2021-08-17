#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:30:06 2020

@author: bene
"""

# Example file to show laplacian edge-detection in python using Imjoy

import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading image
mysize = 750
myblurkernel = 15 # must be odd
myimage = plt.imread('20200120-MEAS-Custom--1.jpg',)
myimage = np.squeeze(myimage[:,:,1])
mycenter_x, mycenter_y = (myimage.shape[0]//2, myimage.shape[1]//2)
myimage = myimage[mycenter_x-mysize:mycenter_x+mysize, mycenter_y-mysize:mycenter_y+mysize]

# remove noise
myimage_blur = cv2.GaussianBlur(myimage,(myblurkernel,myblurkernel),0)

# convolute with proper kernels
laplacian = cv2.Laplacian(myimage_blur,cv2.CV_64F)
sobelx = cv2.Sobel(myimage_blur,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(myimage_blur,cv2.CV_64F,0,1,ksize=5)  # y

mygradient = np.sqrt(sobelx**2+sobely**2)
myedge = mygradient>(np.max(mygradient)*.3 )


#%%
plt.subplot(2,3,1),plt.imshow(myimage,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(mygradient,cmap = 'gray')
plt.title('mygradient'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(myedge,cmap = 'gray')
plt.title('myedge'), plt.xticks([]), plt.yticks([])

plt.show()