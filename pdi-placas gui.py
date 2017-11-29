# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
import cv2
import glob, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

#Para la GUI
from Tkinter import *
import tkFileDialog as filedialog

interface = Tk()


def openfile():
    return filedialog.askopenfilename()


rutaImagen = openfile()


button = Button(interface, text="Open", command=openfile)  # <------
button.grid(column=1, row=1)

img=mpimg.imread(rutaImagen)
imgplot = plt.imshow(img)
plt.show()

OFSET_MIN_AREA = 2.1
OFSET_MAX_AREA = 5.9
OFSET_MIN_WIDTH = 12.0
OFSET_MAX_WIDTH = 14.0
OFSET_MIN_HEIGHT = 60
OFSET_MAX_HEIGHT = 55
OFSET_MIN_Y0 = 20.5
OFSET_X0_WORD_1 = 81.7
OFSET_X0_WORD_2 = 67.7
OFSET_X0_WORD_3 = 53.4
OFSET_X0_NUM_1 = 33.5
OFSET_X0_NUM_2 = 19.5
OFSET_X0_NUM_3 = 5
OFSET_X0 = [OFSET_X0_WORD_1, OFSET_X0_WORD_2, OFSET_X0_WORD_3, OFSET_X0_NUM_1, OFSET_X0_NUM_2, OFSET_X0_NUM_3]
folder = '/home/jeison/Documentos/pdi/proyecto/PDIPlacas/carros/'
folderMuestras = '/home/jeison/Documentos/pdi/proyecto/PDIPlacas/muestras/'
os.chdir(folder)
files = glob.glob('*')
def detect(c):
    ar = 0.0
    shape = "unidentified";
    perim = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * perim, True)
    if len(approx) == 4:
        (x,y,w,h)=cv2.boundingRect(approx)
        ar = float(w)/float(h)
    shape = "square" if ar >= 0.95 and ar <= 1 else "rectangle"
    return shape
def retornaplaca(img_bin, img_resized, contours):
    mayor = 0;
    for h,cnt in enumerate(contours):
        mask = np.zeros(img_bin.shape,np.uint8)#mascara de negros
        cv2.drawContours(mask,[cnt],0,255,-1)
        area = cv2.contourArea(cnt)
        if area>mayor:
            mayor=area
    for h,cnt in enumerate(contours):
        mask = np.zeros(img_bin.shape,np.uint8)#mascara de negros
        cv2.drawContours(mask,[cnt],0,255,-1)
        area = cv2.contourArea(cnt)
        if area > 7000 :
            if area < 150000:
                form = detect(cnt)
                if area == mayor and form == 'rectangle':
                    x,y,w,h = cv2.boundingRect(cnt)
                    box = img_resized[y:y+h,x:x+w]
                    return box
#for file in files:
    #img = cv2.imread(folder+file)

img = cv2.imread(rutaImagen)
img_resized = img[800:2050, 1100:2200]
img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
img_red = img_lab[:, :, 2]
_,img_bin = cv2.threshold(img_red.copy(),140,255,cv2.THRESH_BINARY)

#contours, hierarchy = cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#(_, cnts, _) = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image, contours, hierarchy = cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

box = retornaplaca(img_bin, img_resized, contours)
img = box
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_red = img_lab[:, :, 2]
_,img_bin = cv2.threshold(img_red,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
height, width, channels = img.shape
totalArea = height*width
image, contours, hierarchy = cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
chars = []
features = {'promX': 0, 'promW': 0, 'promY': 0, 'promH': 0}
for h,cnt in enumerate(contours):
    mask = np.zeros(img_bin.shape,np.uint8)
    mask=cv2.resize(mask,(400,400))
    cv2.drawContours(mask,[cnt],0,255,-1)
    area = cv2.contourArea(cnt)
    percentArea = round((area/totalArea)*100, 2)
    if (percentArea > OFSET_MIN_AREA and percentArea < OFSET_MAX_AREA):
        x,y,w,h = cv2.boundingRect(cnt)
        if (w/width*100) < (OFSET_MAX_WIDTH+7):
            percentWidth = (w/width)*100
            box = img[y:y+h, x:x+w]
            features['promX'] += x
            features['promY'] += y
            features['promW'] += w
            features['promH'] += h
            chars.append({'img': box, 'x': (x/width)*100, 'box': {'x':x,'y':y,'w':w,'h':h}})
            cv2.waitKey(0)
imgLen = len(chars)
if imgLen > 0:
    features['promX'] = features['promX']/imgLen
    features['promY'] = features['promY']/imgLen
    features['promW'] = features['promW']/imgLen
    features['promH'] = features['promH']/imgLen
else:
    features['promY'] = (OFSET_MIN_Y0-5)/100*height
    features['promW'] = (OFSET_MIN_WIDTH + OFSET_MAX_WIDTH)/2/100*width
    features['promH'] = (OFSET_MIN_HEIGHT + OFSET_MAX_HEIGHT)/2/100*height
charPositions = [None,None,None,None,None,None]
fixOfset = OFSET_MIN_WIDTH/2
for char in chars:
    if (OFSET_X0_WORD_1 - fixOfset) <= char['x'] <= (OFSET_X0_WORD_1 + fixOfset):
        charPositions[0] = [char['img'], char['box']]
    elif (OFSET_X0_WORD_2 - fixOfset) <= char['x'] <= (OFSET_X0_WORD_2 + fixOfset):
        charPositions[1] = [char['img'], char['box']]
    elif (OFSET_X0_WORD_3 - fixOfset) <= char['x'] <= (OFSET_X0_WORD_3 + fixOfset):
        charPositions[2] = [char['img'], char['box']]
    elif (OFSET_X0_NUM_1 - fixOfset) <= char['x'] <= (OFSET_X0_NUM_1 + fixOfset):
        charPositions[3] = [char['img'], char['box']]
    elif (OFSET_X0_NUM_2 - fixOfset) <= char['x'] <= (OFSET_X0_NUM_2 + fixOfset):
        charPositions[4] = [char['img'], char['box']]
    elif (OFSET_X0_NUM_3 - fixOfset) <= char['x'] <= (OFSET_X0_NUM_3 + fixOfset):
        charPositions[5] = [char['img'], char['box']]
for i in range(0, 6):
    if charPositions[i] == None:
        x = int(OFSET_X0[i]/100*width)
        y = int(features['promY'])
        w = int(OFSET_X0[i]/100*width+features['promW'])
        h = int(features['promY']+features['promH'])
        # charPositions[i] = [ img_bin[y:h, x:w], {'x':x,'y':y,'w':w-x,'h':h-y} ]
        charPositions[i] = [ img_lab[y:h, x:w], {'x':x,'y':y,'w':w-x,'h':h-y} ]
# compareH = cv2.cvtColor(compareH, cv2.COLOR_BGR2GRAY)
placa = ''
fileMatch = ''
os.chdir(folderMuestras)
compFiles = glob.glob('*')
for char in charPositions:
    fileMatch = ''
    imLower = None
    imBigger = None
    lower = 9999999999999
    bigger = -9999999999999
    for compFile in compFiles:
        imComp = cv2.imread(folderMuestras + compFile)
        # clone = char[0].copy()
        # print char[1]
        y = char[1]['y']
        h = y + char[1]['h']
        x = char[1]['x']
        w = x + char[1]['w']
        clone = img_bin[y:h, x:w].copy()
        compH, compW = imComp.shape[:2]
        clone = cv2.resize(clone, (compW, compH), interpolation = cv2.INTER_CUBIC)
        # value = ssim(clone, imComp[:, :, 2], multichannel=True)
        value = 0
        # value = image_similarity_vectors_via_numpy(Image.fromarray(clone), Image.fromarray(imComp))
        # print compFile.split('.')[0], value
        # props = regionprops(clone)
        if value < lower:
            lower = value
            # imLower = img[char[1]['y']:char[1]['y']+char[1]['h'],char[1]['x']:char[1]['x']+char[1]['w']]
        if value > bigger:
            bigger = value
            # imBigger = img[char[1]['y']:char[1]['y']+char[1]['h'],char[1]['x']:char[1]['x']+char[1]['w']]
            fileMatch = compFile
    placa += fileMatch.split('.')[0]
gs = gridspec.GridSpec(1, 7,width_ratios=[1,1,1,1,1,1,1],height_ratios=[1])
plt.subplot(gs[0]), plt.imshow(img), plt.title(placa)
plt.axis("off")
plt.subplot(gs[1]), plt.imshow(charPositions[5][0], cmap = plt.get_cmap('gray')), plt.title('')
plt.axis("off")
plt.subplot(gs[2]), plt.imshow(charPositions[4][0], cmap = plt.get_cmap('gray')), plt.title('')
plt.axis("off")
plt.subplot(gs[3]), plt.imshow(charPositions[3][0], cmap = plt.get_cmap('gray')), plt.title('')
plt.axis("off")
plt.subplot(gs[4]), plt.imshow(charPositions[2][0], cmap = plt.get_cmap('gray')), plt.title('')
plt.axis("off")
plt.subplot(gs[5]), plt.imshow(charPositions[1][0], cmap = plt.get_cmap('gray')), plt.title('')
plt.axis("off")
plt.subplot(gs[6]), plt.imshow(charPositions[0][0], cmap = plt.get_cmap('gray')), plt.title('')
plt.axis("off")
#img=mpimg.imread(rutaImagen)
#imgplot = plt.imshow(img)
plt.show()

# cv2.imshow('Caracteristicas'+' '+      file, box)
# cv2.waitKey(0)

cv2.destroyAllWindows()
