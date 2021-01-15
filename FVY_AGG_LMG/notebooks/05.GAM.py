#!/usr/bin/env python
# coding: utf-8

# ## 05.GAM

# - Realizado por: 
#   Francisco del Val Yague, Alejandro Girón y Laura Martínez González de Aledo
# 
# - Emails: 
#   *francisco.delval@cunef.edu, a.garciagiron@cunef.edu y l.martinezg@cunef.edu*
#   
# Colegio Universitario de Estudios Financieros. CUNEF

#      

# In[1]:


# LIBRERIAS

# Manipulacion del DataFrame
import pandas as pd
import numpy as np

# Modelo GAM
from pygam import LinearGAM, s, f
from pygam.datasets import wage
from sklearn.model_selection import train_test_split
from sklearn import model_selection

# Metricas
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

# Graficos
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Warnings
import warnings;
warnings.filterwarnings('ignore')

# Sin puntos suspensivos
pd.options.display.max_columns = None 
pd.options.display.max_rows = None


#    

# In[2]:


#Lectura de datos

consumo_energia = pd.read_csv("../data/02_intermediate/datos_2016-2020.csv")
consumo_energia = consumo_energia.set_index('time')


# In[3]:


#Encabezado de los datos

consumo_energia.head()


# In[4]:


#cola de la base de datos
consumo_energia.tail()


# In[5]:


#En el grafico se observa que con el aumento de la temperatura aumenta el consumo de energía, evidencia lógica.

consumo_energia.plot(figsize=(15,7))
plt.show()


# In[6]:


#Separación del dataset

X = np.array(consumo_energia.drop(['demanda_energia'],1))
y = np.array(consumo_energia['demanda_energia'])


# In[7]:


#Separamos la muestra de entrenamiento y la de test

validation_size = 0.2
seed = 12345
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


# ### Modelos aditivos generalizados (GAM)

# Un modelo lineal es deseable porque es simple de ajustar, se entiende facilmente, y existen muchas tecnicas disponibles para contrastar las hipotesis del modelo. Sin embargo, en muchos casos, los datos no estan relacionados de forma lineal, por lo que no tiene sentido utilizar los modelos de regresion lineal.
# 
# Ejemplo gráfico:

# ![Captura%20de%20pantalla%202021-01-11%20a%20las%2017.05.01.png](attachment:Captura%20de%20pantalla%202021-01-11%20a%20las%2017.05.01.png)

# #### Construimos el modelo:

# In[8]:


model = LinearGAM(n_splines=10)


# * Ajustamos el modelo a nuestra base de datos de entrenamiento:

# In[9]:


model.gridsearch(X_train, y_train)


# #### Predicción

# In[10]:


#Predicción del modelo

y_pred_validation = model.predict(X_validation)
y_pred_validation


# #### Evaluación de nuestro modelo:

# In[11]:


# diseñamos función para evaluar

def mean_absolute_percentage_error(y_train, y_pred_validation): 
    return np.mean(np.abs((y_train - y_pred_validation) / y_train)) * 100


# In[12]:


def evaluacion_prediccion(consumo_energia, y_pred_validation):
    results = pd.DataFrame({'r2_score':r2_score(consumo_energia, y_pred_validation),
                           }, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(consumo_energia, y_pred_validation)
    results['median_absolute_error'] = median_absolute_error(consumo_energia, y_pred_validation)
    results['mse'] = mean_squared_error(consumo_energia, y_pred_validation)
    results['msle'] = mean_squared_log_error(consumo_energia, y_pred_validation)
    results['mape'] = mean_absolute_percentage_error(consumo_energia, y_pred_validation)
    results['rmse'] = np.sqrt(results['mse'])
    return results


# In[13]:


evaluacion_prediccion(y_validation, y_pred_validation)


# En este modelo podemos ver que el coeficiente de determiación R cuadrado, que explica la cantidad de varianza explicada de los datos, es sustancialmente bajo.
# 
# 
# El mean_absolute_error mide la magnitud media de los errores en un conjunto de pronósticos, sin tener en cuenta su dirección. Mide la precisión de las variables continuas y el rmse, que es el error cuadrático medio que es  la diferencia entre la previsión y los valores observados correspondientes  al cuadrado.
# 
# 
# Ambos se pueden utilizar juntos para diagnosticar la variación en los errores en un conjunto de previsiones. El cuadrático siempre será mayor o igual al MAE; cuanto mayor sea la diferencia entre ellos, mayor será la varianza en los errores individuales de la muestra. 
# 
# La diferencia media entre la predicción del consumo de energía y la observada fue de 0.028.

#  

# ### Referencias
# 
# _https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html_
# 
# 
# _https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515_

#       

#   
