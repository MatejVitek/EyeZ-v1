## This file generates joint image for each folder where patches are
## (see the bottom of this file for function calls)

import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.misc import imresize, imsave #, imread, imsave
from scipy.ndimage import imread

SIGMA = 1.5
#MODE = 1 # 1 = GAUSSIAN filter & gaussian weighted | 0 = flat | -1 = when only local gaussian is applied but not taking into the account when joining

def generateGaussianMask(size, isRGB=True, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is stddev (effective radius).
    """
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2 # integer division (Py.3 uses //)
    else:
        x0 = center[0]
        y0 = center[1]
    
    sigma = size / 5.0 # before size / 3.0 when 2 was not included in Gaussian
    mask = np.exp(-((x-x0)**2 + (y-y0)**2) / (SIGMA*sigma)**2)
    if isRGB:
        mask = np.stack((mask,)*3, axis=-1)

    #plt.imshow(mask)
    #plt.show()
    return mask

def joinPatches(dirSource, imgTargetPrefix, mask, isRGB=True, targetSize=5000, DEBUG=False):
    dictI = {}
    dictW = {}
    pathlist = Path(dirSource).glob('*.png')
    
    for path in pathlist:
        p = os.path.splitext(os.path.basename(str(path)))[0]
        pArr = p.split('_')
        id = str(pArr[0])
        x = int(pArr[1][1:])
        y = int(pArr[2][1:])
        
        #img = cv2.imread(str(path), 0)
        img = imread(str(path), mode='L')#[:, :, ::-1]
        
        h, w = img.shape
        
        if id not in dictI:
            dictI[id] = np.zeros([targetSize,targetSize],dtype=np.int32)
            dictW[id] = np.zeros([targetSize,targetSize],dtype=np.float32)

        #if DEBUG:
            # break the image - for the doggo experiment, so that we see where the patches are:
        #    img = img * np.random.rand(1,1,3)

        img = img * mask
        dictI[id][y:y+h, x:x+w] = dictI[id][y:y+h, x:x+w] + img
        dictW[id][y:y+h, x:x+w] = dictW[id][y:y+h, x:x+w] + mask
        
    for key in dictI.keys():
        img = dictI[key] / dictW[key]

        #img = (img / np.max(img[:]) ) * 255.0
        #if DEBUG:
        #    img = img * SIGMA
        cv2.imwrite(imgTargetPrefix + key + '.png', img)
        #imsave(imgTargetPrefix + key + '.png', img )

#joinPatches('toy', 'test_', generateGaussianMask(41, isRGB=True), isRGB=True, targetSize=300, DEBUG=False)
if __name__ == '__main__':

    joinPatches('./data/Segnet_Predicted_Label_full2/', 'test_', generateGaussianMask(41, isRGB=False), isRGB=False, targetSize=5000)
#joinPatches('../SegNet/Jobs/s/InputData/testannot', 'testannot.png', generateGaussianMask(41, isRGB=False), isRGB=False, targetSize=6000)
