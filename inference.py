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

# parse system arguments
model_location = '../pretrained/fullyConvMedian.hdf5'
img_src = '../data/test/rdata4.bmp'
if len(sys.argv) > 1:
    model_location = str(sys.argv[1])
if len(sys.argv) > 2:
    img_location = str(sys.argv[2]) 
print('Inference on {} using model {}'.format(img_location, model_location))

# load well-learned model
model = load_model(
        model_location,
        custom_objects={
            'tf':tf, 
            'find_medians': find_medians,
            'merge': merge
            })

# load src image as gray-scale image
src_img = cv2.imread(img_location, cv2.IMREAD_GRAYSCALE)
img = np.asarray(src_img / 255.0, np.float)
noisy_img = skimage.util.random_noise(img, mode='s&p', amount=0.7)
noisy_img = cv2.merge((noisy_img, noisy_img, noisy_img))
gx0 = np.reshape(noisy_img, (1, *noisy_img.shape))
Y = model.predict(gx0, verbose=1)
result = np.asarray(Y[0,:,:,:], np.float)
# normalize images
result = cv2.normalize(result, 0, 255, norm_type=cv2.NORM_MINMAX)
noisy_img = cv2.normalize(noisy_img, 0, 255, norm_type=cv2.NORM_MINMAX)
noisy_img = noisy_img[:,:,0]
result = result[:,:,0]

# write images
cv2.imwrite('noisy.bmp', noisy_img)
cv2.imwrite('ret.bmp', result)
print('mse proposed', compare_mse(src_img, result))

