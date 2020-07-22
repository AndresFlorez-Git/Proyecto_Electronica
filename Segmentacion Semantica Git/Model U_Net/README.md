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


El modelo recibe como entrada una imagen de 200x200 pixeles en formato RGB (por lo que tiene 3 canales), primero, pasa por 2 capas convolucionales para obtener 16 imágenes filtradas de tamaño 200x200 (c1 en la figura).
Segundo, dichas 16 imágenes pasan por la capa max pooling, lo cual reduce su dimensionalidad a 100x100, seguido a esto, se pasa por 2 capas convolucionales para obtener 32 imágenes filtradas de tamaño 100x100 (c2 en la figura).
Tercero, dichas 32 imágenes pasan por la capa max pooling, lo cual reduce su dimensionalidad a 50x50, seguido a esto, se pasa por 2 capas convolucionales para obtener 64 imágenes filtradas de tamaño 50x50 (c3 en la figura).
Cuarto, dichas 64 imágenes pasan por la capa max pooling, lo cual reduce su dimensionalidad a 25x25, seguido a esto, se pasa por 2 capas convolucionales para obtener 128 imágenes filtradas de tamaño 25x25 (c4 en la figura).
