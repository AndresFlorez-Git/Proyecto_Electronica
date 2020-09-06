# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 15:37:26 2020

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
############################  Path Base  #####################################
##############################################################################

BASE_PATH = str(os.path.dirname(os.path.abspath(''))) # path del la carpeta Segmentacion Semantica Git

if ('\\' in BASE_PATH) == True:
    separator_dir = '\\'
else:
    separator_dir = '/'



##############################################################################
####################  Path del data set original #############################
##############################################################################

# TRAIN_PATH_IMAGES = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Con ajuste' + separator_dir + 'Images'
# TRAIN_PATH_MASKS = BASE_PATH + separator_dir + 'Data Set' + separator_dir + 'Mezcla' + separator_dir + 'Con ajuste' + separator_dir + 'Masks'


##############################################################################
####################  Path del data set aumentado ############################
##############################################################################

# Path de las imagenes, mascaras e imagenes de prueba
TRAIN_PATH_IMAGES = BASE_PATH + separator_dir + 'Augmented Train Data' + separator_dir + 'Images'
TRAIN_PATH_MASKS = BASE_PATH + separator_dir + 'Augmented Train Data' + separator_dir + 'Masks'



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

    # if name == 'N_C_F':
    #     Y_train_class[n,0] = 1
    #     Y_train_class[n,1] = 0
    #     Y_train_class[n,2] = 0
    #     Y_train_class[n,3] = 0
    # if name == 'N_C_S':
    #     Y_train_class[n,0] = 0
    #     Y_train_class[n,1] = 1
    #     Y_train_class[n,2] = 0
    #     Y_train_class[n,3] = 0
    # if name == 'N_C_T':
    #     Y_train_class[n,0] = 0
    #     Y_train_class[n,1] = 0
    #     Y_train_class[n,2] = 1
    #     Y_train_class[n,3] = 0
    # if name == 'N_S_S':
    #     Y_train_class[n,0] = 0
    #     Y_train_class[n,1] = 0
    #     Y_train_class[n,2] = 0
    #     Y_train_class[n,3] = 1    





# Es necesario organizar de forma aleatoria el dataset aumentado para un correcto entrenamiento del modelo
index = np.arange(X_train.shape[0])
np.random.shuffle(index)


X_train = X_train[index]
Y_train_class = Y_train_class[index]



##############################################################################
#####################  Arquitectura del modelo ###############################
##############################################################################

model = tf.keras.models.Sequential()
model.add(Conv2D(16, (5,5), input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), activation='relu', padding='same'))
model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(32, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (5,5),  activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization(axis = 1))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
#model.add(Dense(200,  activation='relu'))
model.add(Dense(4,  activation='softmax'))
sgd = SGD(lr = 0.01, momentum= 0.9, nesterov=True )

# Salida
model.compile(optimizer='SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

#'rmsprop'

##############################################################################
##########################  Modelcheckpoint ##################################
##############################################################################


checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_pv.h5', verbose=1, save_best_only = True)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=6, monitor='val_loss'),
              tf.keras.callbacks.TensorBoard(log_dir='logs')]

##############################################################################
####################  Resultados del modelo ##################################
##############################################################################
results = model.fit(X_train, Y_train_class, validation_split=0.1, batch_size=8, epochs=50, callbacks=callbacks)

# Descomentar si se quiere guardar el modelo entrenado
Val_acc = results.history['val_acc'][-1]
model.save(BASE_PATH + separator_dir + 'Modelos_Guardados' + separator_dir + 'Modelo_Clasificacion_val_acc_'+str(round(Val_acc,4))+'.h5')

plt.figure(1,figsize=(5,5))
plt.plot(results.history['loss'], label = 'Train')
plt.plot(results.history['val_loss'], label = 'Validation loss')
plt.axis([0,30,0,7])
plt.legend()
plt.show()

plt.figure(2,figsize=(5,5))
plt.plot(results.history['acc'], label = 'Train acc')
plt.plot(results.history['val_acc'], label = 'Validation acc')
plt.legend()
plt.axis([0,30,0,1])
plt.show()

##############################################################################
#########  Visualizar algunos resultados de forma aleatoria ##################
##############################################################################

preds_train = model.predict(X_train, verbose=1)
preds_train = (preds_train>0.5).astype(np.uint8)


# # Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train)-1)
imshow(X_train[ix])
plt.show()

print('Predicción: ', preds_train[ix])
print('Real: ', Y_train_class[ix])
# imshow(np.squeeze(Y_train2[ix]))
# plt.show()
# imshow(np.squeeze(preds_train_t[ix]))
# plt.show()





##############################################################################
##############  Función para visualizar matriz de confusión ##################
##############################################################################
# Esta función se encuentra dentro d los métodos de sklearn

def plot_confusion_matrix(cm, classes,
                        normalize=True,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



##############################################################################
########################  Matriz de confusión ################################
##############################################################################


cm = confusion_matrix(y_true=Y_train_class.argmax(axis=1), y_pred=preds_train.argmax(axis=1))
cm_plot_labels = ['Crakcs','Shadows','Dust','Without failure']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')