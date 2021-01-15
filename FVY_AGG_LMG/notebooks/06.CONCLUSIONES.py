#!/usr/bin/env python
# coding: utf-8

# ## 06. Conclusiones

# - Realizado por: 
#   Francisco del Val Yague, Alejandro Girón y Laura Martínez González de Aledo
# 
# - Emails: 
#   *francisco.delval@cunef.edu, a.garciagiron@cunef.edu y l.martinezg@cunef.edu*
#   
# Colegio Universitario de Estudios Financieros. CUNEF

#   

# #### 1. ARIMA

# ![Captura%20de%20pantalla%202021-01-15%20a%20las%2013.31.36.png](attachment:Captura%20de%20pantalla%202021-01-15%20a%20las%2013.31.36.png)

# #### 2. SARIMA

# ![Captura%20de%20pantalla%202021-01-15%20a%20las%2013.33.38.png](attachment:Captura%20de%20pantalla%202021-01-15%20a%20las%2013.33.38.png)

# El criterio de información de Akaike (AIC) es una medida de la calidad relativa de un modelo estadístico, para un conjunto dado de datos. Como tal, el AIC proporciona un medio para la selección del modelo.
# 
# AIC maneja un trade-off entre la bondad de ajuste del modelo y la complejidad del modelo. Se basa en la entropía de información: se ofrece una estimación relativa de la información perdida cuando se utiliza un modelo determinado para representar el proceso que genera los datos.
# 
# AIC no proporciona una prueba de un modelo en el sentido de probar una hipótesis nula, es decir AIC no puede decir nada acerca de la calidad del modelo en un sentido absoluto. Si todos los modelos candidatos encajan mal, AIC no dará ningún aviso de ello.
# 
# El criterio de información bayesiano (BIC) es un criterio para la selección de modelos entre un conjunto finito de modelos. Se basa, en parte, de la función de probabilidad y que está estrechamente relacionado con el criterio anterior (AIC).
# 
# Cuando el ajuste de modelos, es posible aumentar la probabilidad mediante la adición de parámetros, pero si lo hace puede resultar en sobreajuste. Tanto el BIC y AIC resuelven este problema mediante la introducción de un término de penalización para el número de parámetros en el modelo, el término de penalización es mayor en el BIC que en el AIC.

# #### 3. GAM

# ![Captura%20de%20pantalla%202021-01-15%20a%20las%2013.28.09.png](attachment:Captura%20de%20pantalla%202021-01-15%20a%20las%2013.28.09.png)

# En este modelo podemos ver que el coeficiente de determiación R cuadrado, que explica la cantidad de varianza explicada de los datos, es sustancialmente bajo.
# 
# 
# El mean_absolute_error mide la magnitud media de los errores en un conjunto de pronósticos, sin tener en cuenta su dirección. Mide la precisión de las variables continuas y el rmse, que es el error cuadrático medio que es  la diferencia entre la previsión y los valores observados correspondientes  al cuadrado.
# 
# 
# Ambos se pueden utilizar juntos para diagnosticar la variación en los errores en un conjunto de previsiones. El cuadrático siempre será mayor o igual al MAE; cuanto mayor sea la diferencia entre ellos, mayor será la varianza en los errores individuales de la muestra. 
# 
# La diferencia media entre la predicción del consumo de energía y la observada fue de 0.028.

# #### 4. PROCESO GAUSIANO

# ![Captura%20de%20pantalla%202021-01-15%20a%20las%2013.25.56.png](attachment:Captura%20de%20pantalla%202021-01-15%20a%20las%2013.25.56.png)

# En este modelo podemos ver que el coeficiente de determinación R cuadrado, que explica la cantidad de varianza explicada de los datos, es sustancialmente bajo.
# El mean_absolute_error mide la magnitud media de los errores en un conjunto de pronósticos, sin tener en cuenta su dirección. Mide la precisión de las variables continuas y el rmse, que es el error cuadrático medio que es la diferencia entre la previsión y los valores observados correspondientes al cuadrado.
# Ambos se pueden utilizar juntos para diagnosticar la variación en los errores en un conjunto de previsiones. El cuadrático siempre será mayor o igual al MAE; cuanto mayor sea la diferencia entre ellos, mayor será la varianza en los errores individuales de la muestra.
# La diferencia media entre la predicción del consumo de energía y la observada fue de 0.07
# El coeficiente R cuadrado quizas es más adecuado para indicar la varianza explicada en modelos lineales o el proceso gausiano quizas no se adecue a explicar este tipo de datos, por lo que podria funcionar de manera más precisa en otros ámbitos

# In[ ]:




