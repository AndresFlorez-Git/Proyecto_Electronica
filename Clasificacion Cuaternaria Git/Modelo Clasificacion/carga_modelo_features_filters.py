# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 14:29:54 2020

@author: ANDRES FELIPE FLOREZ
"""


##############################################################################
######################  Sección de importaciones #############################
##############################################################################
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Flatten, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread, imshow
from skimage.transform import resize
from tqdm import tqdm
import random
from keras.models import Model
from keras.utils import plot_model, to_categorical
from sklearn.metrics import confusion_matrix
import itertools
from keras.optimizers import SGD
##############################################################################
######################  Declaración de variables #############################
##############################################################################

# Semilla de números alearorios 
seed = 54
np.random.seed(seed)

# Dimensiones de las imagenes RGB.
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_CHANNELS = 3



##############################################################################
####################  Base Path ############################
##############################################################################




BASE_PATH = str(os.path.dirname(os.path.abspath('')))
print(BASE_PATH)

if ('\\' in BASE_PATH) == True:
    separator_dir = '\\'
else:
    separator_dir = '/'
    
##############################################################################
####################  Path del data set original #############################
##############################################################################

TRAIN_PATH_IMAGES = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Con ajuste' + separator_dir + 'Images'
TRAIN_PATH_MASKS = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Con ajuste' + separator_dir + 'Masks'

##############################################################################
#################  Procesar las imagenes aumentadas ##########################
##############################################################################


# Se obtienen los Ids de las imagenes aumentadas.
Train_images_files = next(os.walk(TRAIN_PATH_IMAGES))[2]
Train_masks_files = next(os.walk(TRAIN_PATH_MASKS))[2]



# Se declaran los arrays que almacenan las imagenes
X_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)
Y_train_class = np.zeros((len(Train_images_files), 1), dtype=np.uint8)


# Se inicia el proceso de resizing de las imagenes de entrenamiento y las de prueba
print('Resizing train images')

for n, id_ in tqdm(enumerate(Train_images_files), total=len(Train_images_files)):
    path_image = TRAIN_PATH_IMAGES + '\\'
    path_mask = TRAIN_PATH_MASKS + '\\'
    img = imread(path_image + Train_images_files[n])[:,:,:IMG_CHANNELS]
    X_train[n] = img
    
    mask = imread(path_mask + Train_masks_files[n])
    mask = np.expand_dims(mask, axis=-1)
    Y_train[n] = mask
    X_train[n] = X_train[n]*Y_train[n]
    
    name = id_[:5]
    if name == 'N_C_F':
        Y_train_class[n] = 1
    if name == 'N_C_S':
        Y_train_class[n] = 2

    if name == 'N_C_T':
        Y_train_class[n] = 3

    if name == 'N_S_S':
        Y_train_class[n] = 0
Y_train_class = to_categorical(Y_train_class, num_classes=4)      
    
    
##############################################################################
#####################  Arquitectura del modelo ###############################
##############################################################################

# modelos_Guardados = os.listdir(BASE_PATH + separator_dir + 'Modelos_Guardados')
model = tf.keras.models.load_model(BASE_PATH + separator_dir +'Modelos_Guardados'+separator_dir + 'Modelo_Clasificacion_val_acc_0.795.h5')

# convert_model_lite = tf.lite.TFLiteConverter.from_keras_model_file('Modelo_Clasificacion_val_acc_0.795.h5')
# tflite_model = convert_model_lite.convert()

# with tf.io.gfile.GFile('Clasificacion_cuaternaria.tflite', 'wb') as f:
#   f.write(tflite_model)

converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file('prueba.h5')
tflite_model = converter.convert()



with open('Clasificacion_cuaternaria.tflite', 'wb') as f:
  f.write(tflite_model)

##############################################################################
#########  Visualizar algunos resultados de forma aleatoria ##################
##############################################################################


preds_train = model.predict(X_train, verbose=1)



# Perform a sanity check on some random validation samples
ix = 5
imshow(X_train[ix])
plt.xticks([])
plt.yticks([])
plt.show()
print('Predicción: ', preds_train[ix])
print('Real: ', Y_train_class[ix])

##############################################################################
#########  Visualizar caracteristicas convolucionales y filtros ##############
##############################################################################


# Se obtienen los pesos del modelo entrenado
weights = model.get_weights()




# Capa de entrada
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

#Contraction path

l0 = tf.keras.layers.Conv2D(16, (5,5), activation='relu', padding='same')(inputs)
l1 = tf.keras.layers.BatchNormalization(axis = 1)(l0)
l2 = tf.keras.layers.MaxPooling2D((2,2))(l1)
l3 = tf.keras.layers.Conv2D(32, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal')(l2)
l4 = tf.keras.layers.BatchNormalization(axis = 1)(l3)
l5 = tf.keras.layers.MaxPooling2D((2,2))(l4)
l6 = tf.keras.layers.Conv2D(64, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal')(l5)
l7 = tf.keras.layers.BatchNormalization(axis = 1)(l6)
l8 = tf.keras.layers.MaxPooling2D((2,2))(l7)
l9 = tf.keras.layers.Conv2D(128, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal')(l8)
l10 = tf.keras.layers.BatchNormalization(axis = 1)(l9)
l11 = tf.keras.layers.MaxPooling2D((2,2))(l10)
l12 = tf.keras.layers.Conv2D(256, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal')(l11)
l13 = tf.keras.layers.BatchNormalization(axis = 1)(l12)
l14 = tf.keras.layers.MaxPooling2D((2,2))(l13)
l15 = tf.keras.layers.Flatten()(l14)
l16 = tf.keras.layers.Dense(4,  activation='softmax')(l15)

# model1.compile(optimizer='SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# model1.summary()




# features

conv_layer_index = [0, 3, 6, 9, 12]

# Generate feature output by predicting on the input image




layers = [l0, l3, l6, l9, l12]
test_fig = ix

for n in range(len(layers)):
    model_n = tf.keras.Model(inputs = [inputs], outputs = [layers[n]])
    model_n.set_weights(weights[0:(2*(n+1))])
    preds_test = model_n.predict(X_train, verbose=0)
    plt.figure(n,figsize=(10,10))
    total_fig = np.shape(preds_test)[-1]
    cto = 1
    for i in range(0,total_fig):
        
        plt.subplot(int(np.ceil(np.sqrt(total_fig))),int(np.ceil(np.sqrt(total_fig))),cto)
        plt.imshow(preds_test[test_fig,:,:,i],cmap='gist_gray',vmin=0,vmax=np.max(preds_test[test_fig,:,:,i]))
        plt.axis('off')
        cto +=1
        if cto >total_fig:
            break
        
        
        
        
# filters


layer = 0
    
layer = layer*2    
max_value = np.max(weights[layer][:,:,:,:])
min_value = np.min(weights[layer][:,:,:,:])
print(max_value)
for j in range(np.shape(weights[layer])[2]):
    plt.figure(j,figsize=(10,10), edgecolor='black')
    total_fig = np.shape(weights[layer])[3]
    cto = 1
    if layer ==0:
        if j ==0:
            colormap = 'Reds'
            title = 'Filtros Canal Rojo'
        elif j ==1:
            colormap = 'Greens'
            title = 'Filtros Canal Verde'
        else:
            colormap = 'Blues'
            title = 'Filtros Canal Azul'
    else:
        colormap = 'Greys'
        title = 'Filtros Capa Conv ' + str(layer) + ' char: ' + str(j+1)
    for i in range(0,total_fig):
        plt.subplot(int(np.ceil(np.sqrt(total_fig))),int(np.ceil(np.sqrt(total_fig))),cto)
        
        plt.imshow((weights[layer][:,:,j,i]-min_value)/(max_value-min_value),cmap=colormap,vmin=0,vmax=1)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.suptitle(title, fontsize=32)
        cto +=1
        if cto >total_fig:
            break
