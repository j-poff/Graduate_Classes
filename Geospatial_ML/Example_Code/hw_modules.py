# -*- coding: utf-8 -*-
"""
read, write, edit point cloud data

@author: jaehoon Jung, PhD, OSU

If you would like to use this code for other purposes, please consult your instructor
"""

import numpy as np
import open3d as o3d
import cv2
from skimage import util
from tkinter import filedialog, messagebox
import os
import tkinter as tk
import laspy
from datetime import datetime
import time
import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd

def convertUnit(inputData,inputType):
    if inputType == 'feet2meter':
        outputData = inputData *0.3048
    
    if inputType == 'meter2feet':
        outputData = inputData/0.3048
    return outputData

def getTime():
    now = datetime.now()
    return str(now.year)+'_'+str(now.month)+'_'+str(now.day)+'_'+str(now.hour)+'_'+str(now.minute)+'_'+str(now.second) 

def plotPC(pc,color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,0:3])
    t = scaleData(color)
    t = -(t-1)
    t = np.uint8(255*t)
    t_c = cv2.applyColorMap(t, cv2.COLORMAP_JET)/255
    t_c = t_c[:,0,:]
    pcd.colors = o3d.utility.Vector3dVector(t_c)
    o3d.visualization.draw([pcd])

def plotImage(zi,t_cmap):
    plt.figure()
    # plt.clf()
    plt.imshow(zi, cmap=t_cmap)
    plt.colorbar()
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(False)
    plt.show()
    
def rasterizePC(pc,dim,cellsize,col): 
    if np.sum(dim) == 0:
        dim[0:2] = pc[:,0:2].min(axis=0) #-- min values of x, y
        dim[2:4] = pc[:,0:2].max(axis=0) #-- max values of x, y
    binSizeX = int(np.floor((dim[2] - dim[0])/cellsize) + 1)
    binSizeY = int(np.floor((dim[3] - dim[1])/cellsize) + 1)    
    #-- weights are the sum of the values belonging to the samples falling into each bin
    zi, yi, xi = np.histogram2d(pc[:,1], pc[:,0], bins=(binSizeY,binSizeX), weights=pc[:,col], normed=False) #-- define weights
    counts, _, _ = np.histogram2d(pc[:,1], pc[:,0], bins=(binSizeY,binSizeX))
    counts[counts==0] = 1
    zi = zi / counts
    zi = np.ma.masked_invalid(zi)
    zi.data[zi.mask] = 0
    return zi.data, dim

def readPtcloud(filePath, fileExtension, param):
    if fileExtension == ".txt":
        ptcloud = pd.read_csv(filePath, delimiter = " ") # X, Y, Z, R, G, B, Time, Intensity, label
        ptcloud = ptcloud.to_numpy() 
        ptcloud[:,[3,4,5,6,7,8]] = ptcloud[:,[7,3,4,5,6,8]] # swap columns 
        ptcloud = np.column_stack((ptcloud,np.arange(len(ptcloud)).astype(float))) # point ID
        ptcloud = np.column_stack((ptcloud,ptcloud[:,8])) # label
    #-- LAS file is a standard format developed by the American Society for Photogrammetry and Remote Sensing (ASPRS) (FileInfo.com)
    #-- LAZ file is a compressed .LAS (LIDAR Data Exchange) file. It has been compressed so it can more easily be stored and shared with others (FileInfo.com)
    if (fileExtension == ".las") or (fileExtension == ".laz"): 
        L = laspy.read(filePath)
        ptcloud = np.array((L.x,L.y,L.z,L.intensity,L.red,L.green,L.blue,L.gps_time,L.classification)).transpose()
        ptcloud = np.column_stack((ptcloud,np.arange(len(ptcloud)).astype(float))) # point ID 
        ptcloud = np.column_stack((ptcloud,np.zeros(len(ptcloud)).astype(float))) # place holder for your classification results (label)
    
    if np.max(ptcloud[:,3]) > 255:
        param.max_intensity = 65535
    else:
        param.max_intensity = 255
    if np.max(ptcloud[:,4]) > 255:
        param.max_rgb = 65535
    else:
        param.max_rgb = 255
        
    if param.unit == 'feet': 
        ptcloud[:,:3] = convertUnit(ptcloud[:,:3],'feet2meter')
        
    #-- X, Y, Z, Intensity, R, G, B, GPS Time, Classification, point ID, your label     
    return ptcloud, param  

class structtype():
    pass

def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def estimateNormals(pc,radius,max_nn):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:,0:3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius,max_nn)) #-- default 1, 30
    return np.array(pcd.normals)

def voxel_downSampling(pc, param):
    rcd = np.zeros((pc.shape[0],4))
    minXYZ = np.min(pc[:,:3],axis=0)
    rcd[:,:3] = np.floor((pc[:,:3] - minXYZ)/param.voxelsize)
    idx = (np.max(rcd[:,0])*np.max(rcd[:,1]))*(rcd[:,2]) + np.max(rcd[:,0])*(rcd[:,1]) + rcd[:,0]
    idx -= np.min(idx) # too large index can lead to errors 
    pc = np.column_stack((pc,idx))
    rcd[:,3] = idx
    rcd = rcd.astype(int) # make it faster than float   
    pc_ds = util.unique_rows(rcd) # much faster than np.unique
    pc_ds = pc_ds.astype(float)
    pc_ds[:,:3] = (pc_ds[:,:3] + param.voxelsize/2) * param.voxelsize + minXYZ
        
    return pc, pc_ds

def writeLAS(outName,x,y,z,i,r,g,b,t,p,s,u,f,c,extraByteName,extraByte):
    header = laspy.LasHeader(point_format=3, version="1.2")
    if extraByteName:
        header.add_extra_dim(laspy.ExtraBytesParams(name=extraByteName, type=int))             
    header.offsets = [np.min(x), np.min(y), np.min(z)]
    header.scales = np.array([0.0001, 0.0001, 0.0001]) # round up x, y, z values 
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    if (len(r) > 0):
        las.red = r.astype(int)
    if (len(g) > 0):
        las.green = g.astype(int)
    if (len(b) > 0):
        las.blue = b.astype(int)
    if (len(i) > 0):
        las.intensity = i.astype(int) 
    if (len(p) > 0):
        las.pt_src_id = p.astype(int)
    if (len(t) > 0):
        las.gps_time = t
    if (len(s) > 0):
        las.scan_angle = s(int)
    if (len(u) > 0):
        las.user_data = u.astype(int)
    if (len(f) > 0):
        las.flag_byte = f.astype(int)
    if (len(c) > 0):
        las.raw_classification = c
        
    if (extraByteName) and (len(extraByte) > 0):
        exec('las.' + extraByteName + '= extraByte')

    las.write(outName)


def main(param, edit_processingTime, edit_progress): 
    # %% read point cloud (las, laz, txt)
    root = tk.Tk()
    inputFilePath_Name = filedialog.askopenfilename()
    inputFilePath = os.path.split(inputFilePath_Name)[0]
    root.withdraw()

    start_time = time.time()
    edit_progress.delete(0, "end"); 
    edit_progress.insert(0, "Now Processing... ") 
    edit_progress.update()
    
    fileExtension = os.path.splitext(inputFilePath_Name)[1]
    
    if not inputFilePath_Name: # warning 
        edit_progress.delete(0, "end"); edit_progress.insert(0, "No files detected"); edit_progress.update()   
        messagebox.showwarning("Warning", "No files detected") # messagebox always return something           
    
    pc, param = readPtcloud(inputFilePath_Name,fileExtension,param)

    # %% pre-processing
    
    #-- voxel-based downsampling 
    pc, pc_ds = voxel_downSampling(pc, param) #-- down sampling may be necessary to speed up the process
    
    #-- display points 
    # plotPC(pc[:,:3],pc[:,2])  
    # plotPC(pc_ds[:,:3],pc_ds[:,2])  

    #-- calculate normal    
    #-- In geometry, a normal is a vector that is perpendicular to a given object (wiki)
    normals = estimateNormals(pc_ds,1,30)
    
    #-- angle of incidence 
    #-- the angle between a ray incident and the line perpendicular to the surface (wiki)
    reference_vector = np.array([0, 0, 1])
    angleOfIncidence = np.rad2deg(np.arccos(np.sum((normals*reference_vector),axis=1) 
                      / np.sqrt(np.sum((normals*normals),axis=1))*np.sqrt(np.sum((reference_vector*reference_vector)))))
    idx = (angleOfIncidence < param.angle_threshold) | ((180-param.angle_threshold) < angleOfIncidence)
    pc_ds_hor = pc_ds[idx, :]
    pc_ds_ver = pc_ds[np.invert(idx), :] 
    # plotPC(pc_ds_hor[:,:3],pc_ds_hor[:,2])  
    # plotPC(pc_ds_ver[:,:3],pc_ds_ver[:,2])  
    #-- Returns True when sharing the same index  
    pc_hor = pc[np.isin(pc[:,-1],pc_ds_hor[:,-1]),:] 
    # plotPC(pc_hor[:,:3],pc_hor[:,2])  

    #-- rasterize pointcloud
    dim = np.array([0,0,0,0],'float') #-- minX, minY, maxX, maxY
    im_intensity, dim = rasterizePC(pc_hor, dim, param.cellsize, col=3)
    im_intensity = scaleData(im_intensity)
    
    #-- display image
    # plotImage(im_intensity, 'jet')

    # %% post processing    
    #-- The original point cloud data are segmetned according to their rasterized 2D locations on the intensity image 
    rc = np.floor((pc_hor[:,0:2] - dim[0:2])/param.cellsize)
    rc = rc.astype(int)
    intensity = im_intensity[rc[:,1],rc[:,0]]
    pc_cls = pc_hor[intensity > param.intensity_threshold,:]
    pc[np.isin(pc[:,9],pc_cls[:,9]),10] = 1
    
    
    # %% write outputs 
    if (param.check_LAS == 1) or (param.check_TIF == 1):

        outputFileName = os.path.split(inputFilePath_Name)[1]
        outputFileName = os.path.splitext(outputFileName)[0]
        outputFilePath = inputFilePath + '/2022Fall_CE560_' + getTime() 
        os.makedirs(outputFilePath) # create a new folder to save outputs 

        #-- write TIFF files 
        if param.check_TIF == 1:
            tiff.imwrite(outputFilePath + '/' + outputFileName + '_intensity.tiff', im_intensity) 
    
        #-- write LAS files   
        if param.check_LAS == 1:
            if param.check_extension == 1:
                pc_extension = '.las'
            else:
                pc_extension = '.laz'
        
            e =  np.array([]) 
            writeLAS(outName = outputFilePath + '/' + outputFileName + '_pointcloud' + pc_extension,
                          x = pc[:,0],
                          y = pc[:,1],
                          z = pc[:,2],
                          i = pc[:,3],
                          r = pc[:,4],
                          g = pc[:,5],
                          b = pc[:,6],
                          t = pc[:,7],
                          p = e,
                          s = e,
                          u = e,
                          f = e,
                          c = e,
                          # extraByteName = [],
                          # extraByte = e,
                          extraByteName = 'label',
                          extraByte = pc[:,10].astype(int)
                          )   

    edit_progress.delete(0, "end")
    edit_progress.insert(0, "Estimation Finished") 
    edit_progress.update()            
    edit_processingTime.delete(0, "end"); 
    edit_processingTime.insert(0, "Total processing time: "+str(round(time.time()-start_time,3))+" sec") 
    edit_processingTime.update()
        
    print("\n--- total processing time %.3f seconds ---\n" % (time.time() - start_time))






