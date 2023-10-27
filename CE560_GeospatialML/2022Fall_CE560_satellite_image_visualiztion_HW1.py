# -*- coding: utf-8 -*-
"""
Assignment 1: satellite image visualization 

@author: Jaehoon Jung, PhD, OSU
"""
from osgeo import gdal 
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pytictoc import TicToc 
tt = TicToc()
import os
import sys

# %% functions 

def displayFeatures(data_array, ft_id_list, cmap):
     plt.figure()
     fig = plt.figure(figsize=(10,7))
     
     ax = fig.add_subplot(1,1,1)
     ax1 = fig.add_subplot(2,2,1)
     ax2 = fig.add_subplot(2,2,2)
     ax3 = fig.add_subplot(2,2,3)
     ax4 = fig.add_subplot(2,2,4)
     
     ax.spines['top'].set_color('none')
     ax.spines['bottom'].set_color('none')
     ax.spines['left'].set_color('none')
     ax.spines['right'].set_color('none')
     ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
     ax.set_xlabel('X (pixels)')
     ax.set_ylabel('Y (pixels)')
     
     color0 = ft_id_list[0]
     color1 = ft_id_list[1]
     color2 = ft_id_list[2]
     color3 = ft_id_list[3]
     
     ax1.set_title(feature_list[color0])
     ax2.set_title(feature_list[color1])
     ax3.set_title(feature_list[color2])
     ax4.set_title(feature_list[color3])
     
     p1 = ax1.imshow(data_array[:,:,color0], cmap)
     plt.colorbar(p1)
     ax1.grid(False)
     p2 = ax2.imshow(data_array[:,:,color1], cmap)
     plt.colorbar(p2)
     ax2.grid(False)
     p3 = ax3.imshow(data_array[:,:,color2], cmap)
     plt.colorbar(p3)
     ax3.grid(False)
     p4 = ax4.imshow(data_array[:,:,color3], cmap)
     plt.colorbar(p4)
     ax4.grid(False)
     
     ax1.set_xlim(left=0, right=8500)
     ax2.set_xlim(left=0, right=8500)
     ax3.set_xlim(left=0, right=8500)
     ax4.set_xlim(left=0, right=8500)
     
     ax1.plot(data_array[:,:,color0])
     ax2.plot(data_array[:,:,color1])
     ax3.plot(data_array[:,:,color2])
     ax4.plot(data_array[:,:,color3])
     
     fig.suptitle('Worldview Satellite Image Features')
     plt.show()


def displayHistograms(data_array,ft_id_list, bins):
    # define the 4 graphs within the figure, 2x2 matrix
    plt.figure()
    fig = plt.figure(figsize=(10,7))
    
    ax = fig.add_subplot(1,1,1)
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    #pull the bands you want from the list
    color0 = ft_id_list[0]
    color1 = ft_id_list[1]
    color2 = ft_id_list[2]
    color3 = ft_id_list[3]
    #pull the band information from the array in the function
    hist0 = data_array[:,:,color0]
    hist1 = data_array[:,:,color1]
    hist2 = data_array[:,:,color2]
    hist3 = data_array[:,:,color3]
    #plot each histogram and assign the appropiate title
    ax1.hist(hist0[mask>0], bins, density=True, histtype='bar')
    ax1.set_title(feature_list[color0])
    ax2.hist(hist1[mask>0], bins, density=True, histtype='bar')
    ax2.set_title(feature_list[color1])
    ax3.hist(hist2[mask>0], bins, density=True, histtype='bar')
    ax3.set_title(feature_list[color2])
    ax4.hist(hist3[mask>0], bins, density=True, histtype='bar')
    ax4.set_title(feature_list[color3])
    #show the plot
    fig.suptitle('Worldview Satellite Image Features')
    plt.show()
    
def findGeoIntersection(raster1,raster2,band1_id,band2_id):
    # author: Jamon Van Den Hoek, Nov/03/2014
    # https://sciience.tumblr.com/post/101722591382/finding-the-georeferenced-intersection-between-two
    
    # load data
    band1 = raster1.GetRasterBand(band1_id)
    band2 = raster2.GetRasterBand(band2_id)
    gt1 = raster1.GetGeoTransform()
    gt2 = raster2.GetGeoTransform()
    
    # find each image's bounding box
    # r1 has left, top, right, bottom of dataset's bounds in geospatial coordinates.
    r1 = [gt1[0], gt1[3], gt1[0] + (gt1[1] * raster1.RasterXSize), gt1[3] + (gt1[5] * raster1.RasterYSize)]
    r2 = [gt2[0], gt2[3], gt2[0] + (gt2[1] * raster2.RasterXSize), gt2[3] + (gt2[5] * raster2.RasterYSize)]
    # print('\t1 bounding box: %s' % str(r1))
    # print('\t2 bounding box: %s' % str(r2))
    
    # find intersection between bounding boxes
    intersection = [max(r1[0], r2[0]), min(r1[1], r2[1]), min(r1[2], r2[2]), max(r1[3], r2[3])]
    if r1 != r2:
        # print('\t** different bounding boxes **')
        # check for any overlap at all...
        if (intersection[2] <= intersection[0]) or (intersection[1] <= intersection[3]):
            intersection = None
            print('\t***no overlap***')
            return
        else:
            # print('\tintersection:',intersection)
            left1 = int(round((intersection[0]-r1[0])/gt1[1])) # difference divided by pixel dimension
            top1 = int(round((intersection[1]-r1[1])/gt1[5]))
            col1 = int(round((intersection[2]-r1[0])/gt1[1])) - left1 # difference minus offset left
            row1 = int(round((intersection[3]-r1[1])/gt1[5])) - top1
            
            left2 = int(round((intersection[0]-r2[0])/gt2[1])) # difference divided by pixel dimension
            top2 = int(round((intersection[1]-r2[1])/gt2[5]))
            col2 = int(round((intersection[2]-r2[0])/gt2[1])) - left2 # difference minus new left offset
            row2 = int(round((intersection[3]-r2[1])/gt2[5])) - top2
            
            #print '\tcol1:',col1,'row1:',row1,'col2:',col2,'row2:',row2
            if col1 != col2 or row1 != row2:
                print("*** MEGA ERROR *** COLS and ROWS DO NOT MATCH ***")
            # these arrays should now have the same spatial geometry though NaNs may differ
            array1 = band1.ReadAsArray(left1,top1,col1,row1)
            array2 = band2.ReadAsArray(left2,top2,col2,row2)

    else: # same dimensions from the get go
        col1 = raster1.RasterXSize # = col2
        row1 = raster1.RasterYSize # = row2
        array1 = band1.ReadAsArray()
        array2 = band2.ReadAsArray()
        
    return array1, array2, col1, row1, intersection
    
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# %% feature list    
feature_list = ["CoastalBlue",      #1
                "Blue",             #2   
                "Green",            #3  
                "Yellow",           #4
                "Red",              #5
                "RedEdge",          #6
                "Near_Infrared_1",  #7  
                "Near_Infrared_2"]  #8

# %% read data 
root = tk.Tk() #opens new window where we can direct it to the files.
filePath = filedialog.askopenfilename() #just stores the location of the folder
filePath = os.path.split(filePath)[0]
root.withdraw()   

tt.tic()
ds_mask = gdal.Open(filePath + "/Saipan_Mask_2m.tif")
ds_feature = gdal.Open(filePath + "/Satellite_Image_WV2_2016Feb05_resample_2m.tif")

# %% Convert to array 
features = []
for i, __ in enumerate(feature_list):
    ft, mask, __, __, __ = findGeoIntersection(ds_feature, ds_mask, i+1, 1)
    ft[mask==0] = 0 
    features.append(scaleData(ft))
features = np.array(features)  #turn it into numpy array format. usually want this.
features = np.moveaxis(features,0,-1)  #changes the order of the dimensions (columns)

# %% visualization 
displayFeatures(features, ft_id_list=[0,3,5,7], cmap='jet')
displayHistograms(features,ft_id_list=[0,3,5,7], bins=[x/100 for x in range(0,100)]) #histograms is the intensity between 0 and 1.
sys.exit(0)

tt.toc()



















