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


### Extracción de características.
Dado que las arquitecturas empleadas en el proceso de segmentación semántica hacen uso de redes neuronales convolucionales, resulta llamativo visualizar las características extraídas por la red capa a capa de los procesos de convolución, para así formase una idea de cómo es el proceso de clasificación o segmentación.
Los siguientes ejemplos son algunos resultados de la implementación del modelo [U-Net](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Model%20U_Net):

Se observa como las capas convolucionales extraen determinadas características de la imagen principal:
![c1](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/c1.png)

Se evidencia la detección de patrones geométricos en la imagen como bordes, superficies y sombras, los cuales están determinados por la intensidad de brillo en las imágenes filtradas.

![c3](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/c3.png)

Se observa que a medida que se adentra en la profundidad de la red convolucional, las características extraídas por el modelo corresponden a aspectos con mayor abstracción.

![C5](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/c5.png) 

### Filtros.
Los parámetros optimizados en las capas convolucionales de las arquitecturas empleadas corresponden a los distintos filtros que se emplean en las imágenes para la extracción de características.
El tamaño del filtro y la cantidad de estos, repercuten directamente en las características extraídas en cada capa convolucional. Cada valor de los filtros típicamente corresponde a valores entre -1 y 1.
Los siguientes ejemplos son algunos resultados de la implementación del modelo [U-Net](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Model%20U_Net):

De la primera capa convolucional, como la entrada se trata de una imagen RGB, se cuenta con 3 canales (Rojo, Verde y Azul), por lo que existen filtros asociados a cada canal para la extracción de características.
![rojos](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/Filtros_rojos.png)
![verdes](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/Filtros_verdes.png)
![azules](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/Filtros_azules.png)

### Métrica y Función de costo
Dada la naturaleza del proceso de segmentación semántica en imágenes, una métrica adecuada para evaluar el rendimiento del modelo implementado corresponde a la precisión (accuracy), el cual es computado como la relación entre la cantidad de pixeles correctamente clasificados y la cantidad total de pixeles de la imagen.
Por otra parte, la función de costo más apropiada para el problema de segmentación binaria (Panel o fondo), resulta conveniente que la función a optimizar sea la denominada *Binary crossentropy*, la cual es un caso específico de la función *Categorical Crossentropy* donde se asume solamente dos categorías.
La función de coste es:

![loss](https://github.com/AndresFlorez-Git/Proyecto_Electronica/blob/master/Segmentacion%20Semantica%20Git/Figures%20README/loss.png)

Donde $\hat{y_i}$ corresponde al i-eximo valor de la salida del modelo, y $y_i$ corresponde al valor objetivo.
 

### Explorar:
- Base de Datos Original. [Aquí](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Data%20Set)
- Preprocesamiento/ Función de Data Augmentation. [Aquí](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Preprocessing)
- Base de Datos aumentado. [Aquí](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Augmented%20Train%20Data)
- Modelo U-Net. [Aquí](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Model%20U_Net)
- Modelo Basic. [Aquí](https://github.com/AndresFlorez-Git/Proyecto_Electronica/tree/master/Segmentacion%20Semantica%20Git/Model%20Basic)

### Referencias:
- The MathWorks, Practical Deep Learning Examples, 2019


