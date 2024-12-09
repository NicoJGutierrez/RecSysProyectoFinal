# ProyectoFinal IIC3633: Recomendación Grupal para Juegos de Mesa

Este repositorio contiene diversos jupyter notebooks usados para calcular y graficar pero tiene uno central llamado ```ProyectoJuegosMesa.ipynb``` y se explica su ejecución más abajo, se incluye también un .csv con los grupos sintéticos creados y una carpeta con las recomendaciones hechas para poder calcular métricas

## Para ejecutar el jupyter notebook:
Se puede directamente correr el notebook completo, se debe elegir que tipo de grupos hacer en la primera celda de **Código nuestro**, la variable llamada ```group_similarity_to_create```, en la misma celda aparecen las opciones posibles.

## Para las metricas:
Se corren en un archivo aparte llamado ```metrics.py```, el cual también tiene una variable a elegir, entre dos distintos sampleos de reviews usados. La variable es ```sampling_method ````, está al comienzo del código ejecutable, justo después de las funciones, tiene dos opciones y están escritas en un comentario al lado de esta.
