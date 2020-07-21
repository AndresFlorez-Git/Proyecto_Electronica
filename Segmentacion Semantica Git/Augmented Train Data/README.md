# Data Augmentation

Corresponden a técnicas para aumentar el tamaño del conjunto de datos con el que se cuenta, con el fin de enriquecer el proceso de entrenamiento de los modelos de segmentación.

Dichas técnicas corresponden a rotaciones, reflexiones, translaciones, cambios de perspectiva y aumento de las fotografías de entrada.
## Ejemplo

A partir imágenes originales, es posible obtener un gran volumen de datos a partir de las transformaciones de los datos originales.
![Monalisa](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/mona.png)
![Monalisa2](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/aug_0_3324.png)
![Monalisa3](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/aug_0_5133.png)
![Monalisa4](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/aug_0_7308.png)
## Aplicación Práctica

Para poder utilizar estas técnicas adecuadamente en el proceso de entrenamiento de un modelo de segmentación semántica resulta necesario realizar las mismas transformaciones tanto a las imágenes como de las mascaras binarias de segmentación.

![ejemplo1](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Data%20Set/Images/28.jpg)
![ejemplo1_t1](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Images/aug_51_1140.png)
![ejemplo1_t2](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Images/aug_51_1149.png)
![ejemplo1_t3](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Images/aug_51_1654.png)
![ejemplo2](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Data%20Set/Masks/Label_28.png)
![ejemplo2_t1](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Masks/aug_51_1140.png)
![ejemplo2_t2](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Masks/aug_51_1149.png)
![ejemplo2_t3](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Masks/aug_51_1654.png)


### Explorar:
- Conjunto de datos original. [Aquí](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Data%20Set)
- Función para Data Augmentation. [Aquí](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Preprocessing)