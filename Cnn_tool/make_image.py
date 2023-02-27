import numpy as np
from tqdm import tqdm
import sys

#viewpointによって作成するhitmapの種類を変える
def hitmap(arr, viewpoint=0, xmax=60, xmin=40, ymax=60, ymin=40):
    output_hitmap = []
    xsize = xmax-xmin
    ysize = ymax-ymin
    for i in tqdm(range(int(arr[:,0].max())+1)):
        arr_tmp = arr[:, arr[0]==i]
        image_tmp = np.zeros((xsize, ysize))
        if viewpoint==0:# zhitmap
            x_tmp = arr_tmp[2]
            y_tmp = arr_tmp[3]
        elif viewpoint==1:# xhitmap
            x_tmp = arr_tmp[1]
            y_tmp = arr_tmp[2]
        elif viewpoint==2:# yhitmap
            x_tmp = arr_tmp[1]
            y_tmp = arr_tmp[3]
        for n in range(len(arr_tmp)):
            if x_tmp[n]>=xmin and x_tmp[n]<xmax and y_tmp[n]>=ymin and y_tmp[n]<ymax:
                image_tmp[int(x_tmp[n]-xmin)][int(y_tmp[n]-ymin)] = arr_tmp[4, n]
        output_hitmap.append(image_tmp)
    return np.array(output_hitmap)

#3次元でのhitmapを作成する
def hitmap3D(arr, xmax=60, xmin=40, ymax=60, ymin=40, layermax=40, layermin=0, nofx=0, nofy=0, nofl=0):
    output_hitmap = []
    xsize = xmax-xmin
    ysize = ymax-ymin
    lsize = layermax-layermin
    if nofx==0:
        nofx=xsize
    if nofy==0:
        nofy=ysize
    if nofl==0:
        nofl=lsize
    if xsize%nofx!=0:
        print("xsize Error")
        sys.exit()
    if ysize%nofy!=0:
        print("ysize Error")
        sys.exit()
    if lsize%nofl!=0:
        print("layersize Error")
        sys.exit()
    xtilesize = xsize//nofx
    ytilesize = ysize//nofy
    layersize = lsize//nofl
    for i in tqdm(range(int(arr[0].max())+1)):
        arr_tmp = arr[:, arr[0]==i]
        image_tmp = np.zeros((nofx, nofy, nofl)) # [x, y, layer]
        for n in range(len(arr_tmp)):
            if arr_tmp[2, n]>=xmin and arr_tmp[2, n]<xmax and arr_tmp[3, n]>=ymin and arr_tmp[3, n]<ymax and arr_tmp[1, n]>=layermin and arr_tmp[1, n]<layermax:
                image_tmp[int((arr[2, n]-xmin)//xtilesize)][int((arr[3, n]-ymin)//ytilesize)][int((arr[1, n]-layermin)//layersize)] = arr[4, n]
        output_hitmap.append(image_tmp)
    return np.array(output_hitmap)

#3次元でのhitmapを1Event分作成する
def hitmap3DbyEvent(arr, xmax=60, xmin=40, ymax=60, ymin=40, layermax=40, layermin=0, nofx=0, nofy=0, nofl=0):
    xsize = xmax-xmin
    ysize = ymax-ymin
    lsize = layermax-layermin
    if nofx==0:
        nofx=xsize
    if nofy==0:
        nofy=ysize
    if nofl==0:
        nofl=lsize
    if xsize%nofx!=0:
        print("xsize Error")
        sys.exit()
    if ysize%nofy!=0:
        print("ysize Error")
        sys.exit()
    if lsize%nofl!=0:
        print("layersize Error")
        sys.exit()
    xtilesize = xsize//nofx
    ytilesize = ysize//nofy
    layersize = lsize//nofl
    image = np.zeros((nofx, nofy, nofl)) # [x, y, layer]
    for n in range(len(arr[0])):
        if arr[2, n]>=xmin and arr[2, n]<xmax and arr[3, n]>=ymin and arr[3, n]<ymax and arr[1, n]>=layermin and arr[1, n]<layermax:
            image[int((arr[2, n]-xmin)//xtilesize)][int((arr[3, n]-ymin)//ytilesize)][int((arr[1, n]-layermin)//layersize)] = arr[4, n]
    return image