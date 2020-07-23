# Segmentación Semántica

El proceso de segmentación semántica corresponde a uno de los avances en el campo del Deep Learning, que provee un entendimiento de las características de una imagen a nivel de pixel. En efecto la segmentación semántica asocia cada pixel de una imagen a una cierta categoría o campo semántico [1].
Este tipo de técnicas resulta muy útil en la detección o identificación de objetos en una imagen, por lo que es posible obtener resultados como los mostrados en la figura de abajo.
![Ejemplo](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/ejemplo_segmentacion.png)
## Arquitecturas típicas de modelos de segmentación semántica.
Los modelos o redes de segmentación semántica típicamente se derivan de la utilización de modelos de clasificación por medio de redes neuronales convolucionales.
Sin embargo, las arquitecturas de segmentación presentan no solo un camino de reducción de dimensionalidad, donde se extraen las características principales de la imagen de entrada (Encoder), sino que presenta una estructura de ampliación de dimensionalidad (up-sampling or Decoder) que permite, a partir de las características de la imagen original, generar la misma imagen con cada pixel asignado a una clasificación en concreto [1].
![arc_segmentacion](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/arc_segmentacion.png)

## Aplicación práctica en el proyecto.
Dadas las características del proyecto resulta necesario la identificación de las estructuras de los paneles fotovoltaicos en una imagen RGB, por lo que, la implementación de un modelo de segmentación semántica que clasifique los pixeles de una imagen en 2 categorías (Panel o Fondo), resulta ideal.
![imagen](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Data%20Set/Images/8.jpg)
![mascara](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Data%20Set/Masks/Label_8.png)

Por lo tanto, se estudian dos arquitecturas de segmentación semántica, modelo U-Net y un modelo básico, para así concretar una arquitectura apropiada para el procesamiento en tiempo real de imágenes RGB. 
