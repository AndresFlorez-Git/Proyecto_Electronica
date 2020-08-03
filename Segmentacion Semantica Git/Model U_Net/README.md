# Modelo U-Net

### Características Y Descripción. 
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


### Proceso de entrenamiento y validación.
#### Disposición de base de datos.
Para el entrenamiento del modelo se utilizo la [base de datos aumentado](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data), la cual se divide en dos conjuntos principales, el 90% corresponde a datos de entrenamiento y el 10% restante corresponde a datos de validación. Cabe aclarar que es necesario realizar una reorganización de la base de datos de forma aleatoria antes de realizar la partición de datos de entrenamiento y validación (Teniendo en cuenta que imágenes y mascaras binarias deben presentar el mismo orden). 
#### Métrica y Función de costo
Dada la naturaleza del proceso de segmentación semántica en imágenes, una métrica adecuada para evaluar el rendimiento del modelo implementado corresponde a la precisión (accuracy), el cual es computado como la relación entre la cantidad de pixeles correctamente clasificados y la cantidad total de pixeles de la imagen. Por otra parte, la función de costo más apropiada para el problema de segmentación binaria (Panel o fondo), resulta conveniente que la función a optimizar sea la denominada Binary crossentropy, la cual es un caso específico de la función Categorical Crossentropy donde se asume solamente dos categorías. La función de coste es:

![metrica](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/loss.png)

De acuerdo con la métrica y función de coste utilizados, se obtienen el desempeño mostrado en la figura durante el proceso de entrenamiento.
![accuracy](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Model%20U_Net/Figures%20README/accuracy.png)
![loss](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Model%20U_Net/Figures%20README/loss.png)
#### Parámetros del modelo 
Para obtener el computo de la cantidad total de parámetros del modelo, se debe tener en cuenta el tipo de capa que se este tratando:
En el siguiente cuadro se resume la cantidad de parámetros por capa.

![summary](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Model%20U_Net/Figures%20README/parametros.png)

La cantidad total de parámetros mostrados en la tabla corresponde en su mayoría a los valores de los filtros utilizados en las capas convolucionales sumado a los términos Bias de cada uno de los filtros.

### Extracción de características.
A través del proceso de optimización del modelo, la información que atraviesa los filtros de las capas convolucionales se obtiene las diferentes características que identifica la red neuronal como patrones geométricos o de superficie, lo que se puede evidenciar como imágenes transformadas a partir de la original.

### Filtros

### Pruebas
