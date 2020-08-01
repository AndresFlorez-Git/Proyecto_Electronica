# Modelo U-Net


El modelo aplicado a la implementación del proyecto presenta la siguiente estructura:
![modelo unet](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Model%20U_Net/Figures%20README/U-Net.png)

Como se evidencia en la figura, el modelo aplicado cuenta con:
- 14 capas convolucionales que aplican filtros (3x3) seguidos de la función de activación ReLU.
- 1 capa convolucional final que aplica un filtro (1x1) seguido de la función de activación Sigmoide.
- 3 capas de Max pooling (pool size (2x2), stride = 2).
- 3 capas de convolución transpuesta (También llamada Deconvolución).
- 3 tres procesos de concatenación de características.
- Todas las capas convolucionales cuentan con padding para evitar la reducción de dimensionalidad.

El proceso de segmentación semántica ocurre en dos procesos, contracción y expansión. La contracción consiste en la paulatina reducción de dimensionalidad de los datos de entrada en una serie de características obtenidas por las capas convolucionales. Por otro lado, la expansión consiste en la recuperación de la imagen original para así crear una máscara binaria de segmentación.

El modelo recibe como entrada una imagen de 200x200 pixeles en formato RGB (por lo que tiene 3 canales), primero, pasa por 2 capas convolucionales para obtener 16 imágenes filtradas de tamaño 200x200 (c1 en la figura).
Segundo, dichas 16 imágenes pasan por la capa max pooling, lo cual reduce su dimensionalidad a 100x100, seguido a esto, se pasa por 2 capas convolucionales para obtener 32 imágenes filtradas de tamaño 100x100 (c2 en la figura).
Tercero, dichas 32 imágenes pasan por la capa max pooling, lo cual reduce su dimensionalidad a 50x50, seguido a esto, se pasa por 2 capas convolucionales para obtener 64 imágenes filtradas de tamaño 50x50 (c3 en la figura).
Cuarto, dichas 64 imágenes pasan por la capa max pooling, lo cual reduce su dimensionalidad a 25x25, seguido a esto, se pasa por 2 capas convolucionales para obtener 128 imágenes filtradas de tamaño 25x25 (c4 en la figura).
A partir de este punto, comienza el segmento del modelo de expansión, donde se reconstruyen las características de la imagen original a partir de la información de las capaz del segmento del modelo de contracción.
Por lo tanto, el quinto paso, a partir de la información en c4, las 128 imágenes de tamaño 25x25 se pasan por un proceso de convolución transpuesto para obtener 64 imágenes de tamaño 50x50 a las cueles se le añaden las imágenes obtenidas en C3, por lo que se obtiene un conjunto de 128 imágenes las cuales pasan por dos capas convolucionales para así obtener 64 imágenes de tamaño 50x50 (c5 en la figura).
Sexto, dichas 64 imágenes de tamaño 50x50 pasan por un proceso de convolución transpuesto para obtener 32 imágenes de tamaño 100x100 a las cueles se le añaden las imágenes obtenidas en C2, por lo que se obtiene un conjunto de 64 imágenes las cuales pasan por dos capas convolucionales para así obtener 32 imágenes de tamaño 100x100 (c6 en la figura).
Séptimo, dichas 32 imágenes de tamaño 100x100 pasan por un proceso de convolución transpuesto para obtener 16 imágenes de tamaño 200x200 a las cueles se le añaden las imágenes obtenidas en C2, por lo que se obtiene un conjunto de 32 imágenes las cuales pasan por dos capas convolucionales para así obtener 16 imágenes de tamaño 200x200 (c7 en la figura).
Por último, Dichas 16 imágenes de tamaño 200x200 se pasa por una capa de convolución final con un filtro de tamaño 1x1 y función de activación sigmoide para obtener 1 imagen de tamaño 200x200, donde en cada pixel se obtiene una probabilidad de pertenecer a la categoría panel solar.  



