# Aumentar datos

Corresponden a técnicas para aumentar el tamaño del conjunto de datos con el que se cuenta, con el fin de enriquecer el proceso de entrenamiento de los modelos de segmentación.

Dichas técnicas corresponden a rotaciones, reflexiones, translaciones, cambios de perspectiva y aumento de las fotografías de entrada.
## Ejemplo

A partir imágenes originales, es posible obtener un gran volumen de datos a partir de las transformaciones de los datos originales.
![Monalisa](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/monalisa.jpg)
![Arrow](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/arrow.png)
![Monalisa2](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/aug_0_3324.png)
![Monalisa3](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/aug_0_5133.png)
![Monalisa4](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data/Examples/aug_0_7308.png)
## Aplicación Práctica

Para poder utilizar estas técnicas adecuadamente en el proceso de entrenamiento de un modelo de segmentación semántica resulta necesario realizar las mismas transformaciones tanto a las imágenes como de las mascaras binarias de segmentación.
