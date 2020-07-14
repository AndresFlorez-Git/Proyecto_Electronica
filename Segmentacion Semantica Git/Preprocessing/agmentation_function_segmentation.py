# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:00:32 2020

@author: ANDRES FELIPE FLOREZ
"""
import keras
from skimage.io import imread, imshow, imsave
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import shutil

# Este programa almacena la función augmetation_data_set, la cual se encarga de crear el aumento de fotografias a partir de transformaciones de imagenes entregadas como
# input.  Dicho aumento de datos queda almacenado en la carpeta llamada Augmentation_data_set_segmentation en el directorio actual de trabajo.


def augmetation_data_set(path_folder_images,path_folder_masks, Size_image, images_per_photo):
    # Se asigna el nombre a los directorios donde se almacenarán los datos aumentados.
    New_path = 'Augmentation_data_set_segmentation'
    New_path_images = 'images'
    New_path_masks = 'masks'
    
    # Se crean los directorios asociados al destino de de los datos aumentados
    if os.path.exists(New_path) == True:
        shutil.rmtree(New_path)
    if os.path.exists(New_path) == False: 
        os.mkdir(New_path)
        os.mkdir(New_path + '\\' + New_path_images)
        os.mkdir(New_path + '\\' + New_path_masks)
    
    # Se usa una semilla especidfica para numeros aleatorios, puede ser cualquier numero entero
    seed = 54
    np.random.seed(seed)

    # Se asignantodas las transformaciónes que se desean realizar a las imagenes.
    datagen_arg = dict(
        rotation_range=360,
        width_shift_range= 0.3,
        height_shift_range=0.1,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'reflect'
        )

    # Se crean los objetos asociados a la descripción de la transformación de las imagenes y las mascarads de la segmentación
    image_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_arg)
    mask_datagen = keras.preprocessing.image.ImageDataGenerator(**datagen_arg)
    
    # Se declaran las listas que almacenarárn cada una de las imagenes del data set original
    data_set_images = []
    data_set_masks = []
    
    # Se obtienen los Ids de las imagenes del data set original
    images_ids = next(os.walk(path_folder_images))[2]
    masks_ids = next(os.walk(path_folder_masks))[2]
    
    
    # Se almacenan las imagenes en las listas declaradas y al mismo tiempo se reescalan al tamaño adecuado que se requiere.
    for n, id_ in tqdm(enumerate(images_ids), total=len(images_ids)):
        img = imread(path_folder_images + '\\' + images_ids[n] )[:,:,:3]
        img = Image.fromarray(img, 'RGB')
        #img = img.resize((Size_image,Size_image))
        img2 = imread(path_folder_masks + '\\' + masks_ids[n] )
        # img2 = Image.fromarray(img2, 'RGB')
        #img2 = img2.resize((Size_image,Size_image))
        data_set_images.append(np.array(img))
        data_set_masks.append(np.array(img2))
    
    # Se convierte la lista en un numpy array
    data_set_images = np.array(data_set_images)
    data_set_masks = np.array(data_set_masks)
    data_set_masks = np.expand_dims(data_set_masks, axis=-1)

    # Los siguientes for loop permiten almacenar las imagenes transformadas en los directorios asignados
    i = 0
    for batch in image_datagen.flow(data_set_images, save_to_dir = New_path + '\\' + New_path_images,
                                                      save_prefix = 'aug',
                                                      save_format = 'png',
                                                      batch_size = 1, seed = seed
                                                      ):
        i +=1
        if i == len(images_ids)*images_per_photo:
            break  
    i = 0
    for batch2 in mask_datagen.flow(data_set_masks, save_to_dir = New_path + '\\' + New_path_masks,
                                                      save_prefix = 'aug',
                                                      save_format = 'png',
                                                      batch_size = 1, seed = seed
                                                      ):
        i +=1
        if i == len(images_ids)*images_per_photo:
            break  
        
        
# augmetation_data_set('Dataset\Segmentar\Entreno\ImaEscalada','Dataset\Segmentar\Entreno\LabelsFinales', 200, 20)
