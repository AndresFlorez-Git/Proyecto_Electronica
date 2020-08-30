# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 12:55:30 2020

@author: ANDRES FELIPE FLOREZ
"""


import keras
from skimage.io import imread, imshow, imsave
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil
from skimage.transform import resize


BASE_PATH = str(os.path.dirname(os.path.abspath(''))) # path del la carpeta Segmentacion Semantica Git

if ('\\' in BASE_PATH) == True:
    separator_dir = '\\'
else:
    separator_dir = '/'
    
    
# Semilla de números alearorios 
seed = 54
np.random.seed(seed)

# Dimensiones de las imagenes RGB.
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_CHANNELS = 3
    
    
path_folder_images = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Sin ajuste' + separator_dir + 'Images'
path_folder_masks = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Sin ajuste' + separator_dir + 'Masks'

path_folder_images_ajuste = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Con ajuste' + separator_dir + 'Images'
path_folder_masks_ajuste = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Con ajuste' + separator_dir + 'Masks'


Train_images_files = next(os.walk(path_folder_images))[2]
Train_masks_files = next(os.walk(path_folder_masks))[2]

print(Train_images_files[0])
print(Train_images_files[0][0:5])
# Se declaran los arrays que almacenan las imagenes
X_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)


# Se inicia el proceso de resizing de las imagenes de entrenamiento y las de prueba
print('Resizing train images')

for n, id_ in tqdm(enumerate(Train_images_files), total=len(Train_images_files)):
    path_image = path_folder_images + '\\'
    path_mask = path_folder_masks + '\\'
    img = imread(path_image + Train_images_files[n])[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    imsave(path_folder_images_ajuste + separator_dir + Train_images_files[n],img)
    X_train[n] = img
    
    mask = imread(path_mask + Train_masks_files[n])
    
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # mask = np.expand_dims(mask, axis=-1)
    imsave(path_folder_masks_ajuste + separator_dir + Train_masks_files[n],mask)
    Y_train[n] = mask

