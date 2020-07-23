# Segmentación Semántica

El proceso de segmentación semántica corresponde a uno de los avances en el campo del Deep Learning, que provee un entendimiento de las caracteristicas de una imagen a nivel de pixel. En efecto la segmentación semantica asocia cada pixel de una imagen a una cierta categoria o campo semantico [1].
Este tipo de tecnicas resulta muy util en la deteccion o identificacion de objetos en una imagen, por lo que es posible obtener resultados como los mostrados en la figura de abajo.

## Arquitecturas tipicas de modelos de segmentación semántica.
Los modelos o redes de segmentacion semantica tipicamente se derivan de la utilizacion de modelos de clasificación por medio de redes neuronales convolucionales.
Sin embargo, las arquitecturas de segmentacion presentan no solo un camino de reduccion de dimenionalidad, donde se extraen las caracteristicas principales de la imagen de entrada (Encoder), sino que presenta una estructura de ampliacion de dimencionalidad (up-sampling or Decoder) que permite, a partir de las caracteristicas de la imagen original,  generar la misma imagen con cada pixel asignado a una clasificación en concreto [1].



