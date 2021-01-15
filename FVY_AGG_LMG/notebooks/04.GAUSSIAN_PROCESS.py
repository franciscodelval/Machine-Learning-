#!/usr/bin/env python
# coding: utf-8

# ## 04.GAUSSIAN_PROCESS

# - Realizado por: 
#   Francisco del Val Yague, Alejandro Girón y Laura Martínez González de Aledo
# 
# - Emails: 
#   *francisco.delval@cunef.edu, a.garciagiron@cunef.edu y l.martinezg@cunef.edu*
#   
# Colegio Universitario de Estudios Financieros. CUNEF

#    

# In[2]:


# LIBRERIAS

# Manipulacion del DataFrame
import pandas as pd
import numpy as np

# Proceso Gaussiano
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

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


# In[3]:


#Cargamos las bases de datos
consumo_energia = pd.read_csv("../data/02_intermediate/datos_2016-2020.csv")
consumo_energia = consumo_energia.set_index('time')


# In[4]:


#Visualizamos el encabezado
consumo_energia.head()


# In[5]:


#Visualizamos el final de la base de datos
consumo_energia.tail()


# In[6]:


#A priori, se observa que con las temperaturas mas altas se consume mas energía
consumo_energia.plot(figsize=(15,7))
plt.show()


# #### Dividimos nuestra base de datos:

# In[7]:


#dividimos en train y en test
X = np.array(consumo_energia.drop(['demanda_energia'],1))
y = np.array(consumo_energia['demanda_energia'])


# In[8]:


#20% para test y 80% Train
validation_size = 0.2
seed = 12345
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


# ### Gaussian Process

# El proceso Gaussiano se trata de un modelo de regresión no paramétrica cuyo objetivo es encontrar una relación sencilla y útil entre X e Y para poder explicar y predecir el valor de la Y a partir de X. El modelo de regresión no pramétrica plantea una estimación de tipo local, que resulta más conveniente en algunas configuraciones de datos.

# #### Construimos el modelo:

# In[9]:


kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))


# In[10]:


model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)


# * Ajustamos el modelo a nuestra base de datos de entrenamiento:

# In[11]:


model.fit(X_train, y_train)


# #### Predicción

# In[12]:


y_pred_validation, std_validation = model.predict(X_validation, return_std=True)


# In[13]:


plt.plot(figsize=(15,7))
plt.errorbar(y_validation, y_pred_validation, yerr=std_validation, fmt='o')
plt.title('Gaussian process regression, R2=%.2f' % r2_score(y_validation, y_pred_validation))
plt.xlabel('Actual')
plt.ylabel('Predicted')


# #### Evaluación de nuestro modelo:

# In[14]:


def mean_absolute_percentage_error(y_train, y_pred_validation): 
    return np.mean(np.abs((y_train - y_pred_validation) / y_train)) * 100


# In[15]:


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


# In[16]:


evaluacion_prediccion(y_validation, y_pred_validation)


# En este modelo podemos ver que el coeficiente de determinación R cuadrado, que explica la cantidad de varianza explicada de los datos, es sustancialmente bajo.
# 
# El mean_absolute_error mide la magnitud media de los errores en un conjunto de pronósticos, sin tener en cuenta su dirección. Mide la precisión de las variables continuas y el rmse, que es el error cuadrático medio que es la diferencia entre la previsión y los valores observados correspondientes al cuadrado.
# 
# Ambos se pueden utilizar juntos para diagnosticar la variación en los errores en un conjunto de previsiones. El cuadrático siempre será mayor o igual al MAE; cuanto mayor sea la diferencia entre ellos, mayor será la varianza en los errores individuales de la muestra.
# 
# La diferencia media entre la predicción del consumo de energía y la observada fue de 0.07
# 
# El coeficiente R cuadrado quizas es más adecuado para indicar la varianza explicada en modelos lineales o el proceso gausiano quizas no se adecue a explicar este tipo de datos, por lo que podria funcionar de manera más precisa en otros ámbitos

#     

# ### Referencias
# 
# * Codigo _https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319_
# 
# 
# * Grafico _https://towardsdatascience.com/getting-started-with-gaussian-process-regression-modeling-47e7982b534d_

#    
