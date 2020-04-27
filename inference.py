import cv2
import sys
import math
import skimage
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
from skimage.measure import compare_psnr, compare_mse

from model import *

'''
Simply running inference on 1 image.
Usage: 
    python inference.py model_location img_location
Args:
    model_location: str, '../pretrained/median5Res32Decrease-185.hdf5' as default
    img_location: str, '../data/BSD300/' as default
'''

# parse system arguments
#model_location = '../pretrained/median5Res32Decrease-185.hdf5'
model_location = '../pretrained/fullyConvMedian.hdf5'
#img_location = '../data/BSD300/102061.jpg'
img_src = '../data/test/rdata4.bmp'
img_location = '../data/test/rdata4.bmp'
if len(sys.argv) > 1:
    model_location = str(sys.argv[1])
if len(sys.argv) > 2:
    img_location = str(sys.argv[2]) 
print('Inference on {} using model {}'.format(img_location, model_location))


model = load_model(
        model_location,
        custom_objects={
            'tf':tf, 
            'find_medians': find_medians,
            'merge': merge
            })

# ori_img = cv2.imread(img_src)
src_img = cv2.imread(img_location, cv2.IMREAD_GRAYSCALE)
# src_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2RGB)
img = np.asarray(src_img / 255.0, np.float)
noisy_img = skimage.util.random_noise(img, mode='s&p', amount=0.7)
noisy_img = cv2.merge((noisy_img, noisy_img, noisy_img))
# noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_GRAY2RGB)
print(noisy_img.shape)
gx0 = np.reshape(noisy_img, (1, *noisy_img.shape))
#gx0 = np.reshape(img, (1, *img.shape))
Y = model.predict(gx0, verbose=1)
result = np.asarray(Y[0,:,:,:], np.float)
#cv2.imshow('original', src_img)
#cv2.imshow('bef', noisy_img)
#cv2.imshow('aft', result)
result = cv2.normalize(result, 0, 255, norm_type=cv2.NORM_MINMAX)
noisy_img = cv2.normalize(noisy_img, 0, 255, norm_type=cv2.NORM_MINMAX)
#noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)
#result = cv2.cvtColor(restult, cv2.COLOR_BGR2GRAY)
noisy_img = noisy_img[:,:,0]
result = result[:,:,0]
cv2.imwrite('noisy.bmp', noisy_img)
cv2.imwrite('ret.bmp', result)
#print('psnr original', compare_psnr(src_img, noisy_img))
#print('psnr smoothed', compare_psnr(src_img, result))
print('mse proposed', compare_mse(src_img, result))
#print('mse noise', compare_mse(img, noisy_img))
#print('mse ori', compare_mse(src_img, src_img))

