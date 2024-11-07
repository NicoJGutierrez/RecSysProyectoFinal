# RecSysProyectoFinal

## Para la entrega 2:
Quiero revisar qué puedo hacer de las siguientes cosas:
- Revisar que hace airbnb para armar listas de estancias para grupos de personas, si su recomendador tiene eso en cuenta.
- Priorización de metadata específico por usuario: Que para cada usuario se pueda manejar que metadata es relavante para él/ella.


## Lo que haremos para nuestro recomendador:
1) Hacer un recomendador simple que entregue los ratings normalizados de cada persona. La idea es generar una matriz completa de los ítems usando:
    - i-knn a modo de base
    - Un recomendador que utilice metadata para generar la matriz completa
2) Tomar las listas de recomendaciones generadas por el sistema para cada usuario y combinarlas en una sola usando las siguientes ponderaciones:
    - (fuente: <https://towardsdatascience.com/an-introduction-to-group-recommender-systems-8f942a06db56>)
    - Utilitario-Aditivo (equivalente a promedio)
    - Utilitario-Multiplicativo
    - Minimización de miseria (maximizar el mínimo puntaje entre los usuarios)
    - Utilitario (multiplicativo o aditivo) sin miseria (se determina un mínimo razonable para cada usuario y se maximizan utilidades luego)
3) Le podríamos preguntar a Chat-GPT qué es lo más importante para que la gente disfrute de un juego de mesa y en función de lo que responda elegir la ponderación.
    - Esto porque probablemente nos va a dar un consenso razonable.
    - Esto se puede hacer para el trabajo si compartimos la sesión de chat.
4) Es muy importante que los rating de cada usuario estén normalizados para que esas recomendaciones sean mejores.
5) Si utilizamos metadata, generar parámetros por usuario para que elijan qué tipo de metadata es más importante para cada uno.
