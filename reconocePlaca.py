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

# Larger CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from matplotlib import pyplot as plt

K.set_image_dim_ordering('th')

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



#plt.imshow(charPositions[0][0])
#plt.show()





# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# load json and create model
json_file = open('/home/jeison/Documentos/pdi/red neuronal/model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/home/jeison/Documentos/pdi/red neuronal/model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#scores = loaded_model.evaluate(X_test, y_test, verbose=0)
#print("Large CNN Error: %.2f%%" % (100-scores[1]*100))








def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva


for index in range(0, 3):
    plt.imshow(charPositions[index][0])
    plt.savefig('/home/jeison/Documentos/pdi/red neuronal/aux.png', bbox_inches='tight', pad_inches=0)#guarda imagen sin procesar
    plt.show()
    img= Image.open("/home/jeison/Documentos/pdi/red neuronal/aux.png")
    enhancer = ImageEnhance.Color(img)
    img= enhancer.enhance(0.0)
    enhancer = ImageEnhance.Brightness(img)
    img= enhancer.enhance(2.0)
    enhancer = ImageEnhance.Contrast(img)
    img= enhancer.enhance(10.0)
    img.save("/home/jeison/Documentos/pdi/red neuronal/aux.png")
    x=[imageprepare("/home/jeison/Documentos/pdi/red neuronal/aux.png")]#file path here
    print(len(x))# mnist IMAGES are 28x28=784 pixels
    print(x[0])
    #Now we convert 784 sized 1d array to 24x24 sized 2d array so that we can visualize it
    newArr=[[0 for d in range(28)] for y in range(28)]
    k = 0
    for i in range(28):
        for j in range(28):
            newArr[i][j]=x[0][k]
            k=k+1

    #for i in range(28):
        #for j in range(28):
            #print(newArr[i][j])
            # print(' , ')
            #print('\n')


    plt.imshow(newArr, interpolation='nearest')
    plt.savefig('MNIST_IMAGE.png')#save MNIST image
    plt.show()#Show / plot that image

    numpyvar = np.array(newArr)
    pr = loaded_model.predict_classes(numpyvar.reshape((1, 1, 28, 28)))
    print(pr)
