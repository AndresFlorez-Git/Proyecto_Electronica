# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:17:25 2020

@author: ANDRES FELIPE FLOREZ
"""



##############################################################################
######################  Sección de importaciones #############################
##############################################################################
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread, imshow
from skimage.transform import resize
from tqdm import tqdm
import random

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
####################  Path del data set original #############################
##############################################################################

# TRAIN_PATH_IMAGES = 'Dataset\Segmentar\Entreno\ImaEscalada'
# TRAIN_PATH_MASKS = 'Dataset\Segmentar\Entreno\LabelsFinales'


# Función que permite crear el aumento de fotografias. Mirar augmentation_function_segmentation.py para más info.
# augmetation_data_set('Dataset\Segmentar\Entreno\ImaEscalada','Dataset\Segmentar\Entreno\LabelsFinales', 200, 20)

##############################################################################
####################  Path del data set aumentado ############################
##############################################################################
BASE_PATH = str(os.path.dirname(os.path.abspath('')))

if ('\\' in BASE_PATH) == True:
    separator_dir = '\\'
else:
    separator_dir = '/'
# Path de las imagenes, mascaras e imagenes de prueba
TRAIN_PATH_IMAGES = BASE_PATH + separator_dir + 'Augmented Train Data' + separator_dir + 'Images'
TRAIN_PATH_MASKS = BASE_PATH + separator_dir + 'Augmented Train Data' + separator_dir + 'Masks'
TEST_PATH_IMAGES = BASE_PATH + separator_dir +'Data Set' + separator_dir +'Test'


##############################################################################
#################  Procesar las imagenes aumentadas ##########################
##############################################################################

# Se obtienen los Ids de las imagenes aumentadas.
Train_images_files = next(os.walk(TRAIN_PATH_IMAGES))[2]
Train_masks_files = next(os.walk(TRAIN_PATH_MASKS))[2]
Test_images_files = next(os.walk(TEST_PATH_IMAGES))[2]


# Se declaran los arrays que almacenan las imagenes
X_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(Train_images_files), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
X_test = np.zeros((len(Test_images_files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)


# Se inicia el proceso de resizing de las imagenes de entrenamiento y las de prueba
print('Resizing train images')

for n, id_ in tqdm(enumerate(Train_images_files), total=len(Train_images_files)):
    path_image = TRAIN_PATH_IMAGES + separator_dir
    path_mask = TRAIN_PATH_MASKS + separator_dir
    img = imread(path_image + Train_images_files[n])[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = imread(path_mask + Train_masks_files[n])
    mask = np.expand_dims(mask, axis=-1)
    Y_train[n] = mask

print('Resizing test images')
for n, id_ in tqdm(enumerate(Test_images_files), total=len(Test_images_files)):
    path_image = TEST_PATH_IMAGES + separator_dir
    img = imread(path_image + Test_images_files[n])[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    
##############################################################################
#####################  Arquitectura del modelo ###############################
##############################################################################

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3,3), input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3),  activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2,2)))
model.add(Conv2DTranspose(64, (3,3), strides=(2,2), activation='relu', padding='same'))
model.add(Conv2DTranspose(32, (3,3), strides=(2,2), activation='relu', padding='same'))
model.add(Conv2D(1, (1,1),  activation='sigmoid', padding='same'))


# Salida
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()



##############################################################################
##########################  Modelcheckpoint ##################################
##############################################################################


checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_pv.h5', verbose=1, save_best_only = True)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
              tf.keras.callbacks.TensorBoard(log_dir='logs')]

##############################################################################
####################  Resultados del modelo ##################################
##############################################################################
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

# Descomentar si se quiere guardar el modelo entrenado
# model.save('Modelo_de_segmentacion2.h5')

plt.figure(1,figsize=(5,5))
plt.plot(results.history['loss'], label = 'Train')
plt.plot(results.history['val_loss'], label = 'Validation loss')
plt.legend()
plt.show()

plt.figure(2,figsize=(5,5))
plt.plot(results.history['acc'], label = 'Train acc')
plt.plot(results.history['val_acc'], label = 'Validation acc')
plt.legend()
plt.show()

##############################################################################
#########  Visualizar algunos resultados de forma aleatoria ##################
##############################################################################


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.9).astype(np.uint8)
preds_val_t = (preds_val > 0.9).astype(np.uint8)
preds_test_t = (preds_test > 0.9).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t)-1)
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t)-1)
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()


# Perform a sanity check on some random validation samples
ix = 0
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()


