#!/usr/bin/env python
# coding: utf-8

# ## 02.ARIMA

# - Realizado por: 
#   Francisco del Val Yague, Alejandro García Girón y Laura Martínez González de Aledo
# 
# - Emails: 
#   *francisco.delval@cunef.edu, a.garciagiron@cunef.edu y l.martinezg@cunef.edu*
#   
# Colegio Universitario de Estudios Financieros. CUNEF

#       

# In[43]:


# LIBRERIAS

# Manipulacion del DataFrame
import pandas as pd
import numpy as np

# Series Temporales
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from pmdarima.arima import auto_arima
import itertools

# Metricas
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error

# Graficos
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import seaborn as sbn
from pylab import rcParams # seasonal_decompose

# Warnings
import warnings;
warnings.filterwarnings('ignore')

# Sin puntos suspensivos
pd.options.display.max_columns = None 
pd.options.display.max_rows = None 


# In[77]:


#Cargamos la base de datos
consumo_energia = pd.read_csv("../data/02_intermediate/demanda_energia_total.csv")


# In[78]:


#Visualizamos el encabezado
consumo_energia.head()


# In[79]:


#Visualizamos el final de la base de datos
consumo_energia.tail()


# In[80]:


#A priori, se observa que con las temperaturas mas altas se consume mas energía
consumo_energia.plot(figsize=(15,7))
plt.show()


# ### Modelo Arima

# Modelo estadístico que utiliza variaciones y regresiones de datos estadísticos con el fin de encontrar patrones para una predicción hacia el futuro. Se trata de un modelo dinámico de series temporales, es decir, las estimaciones futuras vienen explicadas por los datos del pasado y no por variables independientes.

# In[81]:


dates = pd.date_range(start='2016-01-01', freq='MS', periods=len(consumo_energia))
dates[0:60]


# In[82]:


consumo_energia.set_index(dates,inplace=True)
consumo_energia.drop('time',axis=1,inplace=True)


# #### Dividimos nuestra base de datos

# In[83]:


train = (consumo_energia[consumo_energia.index < '2018-06-01'])
test = (consumo_energia[consumo_energia.index >= '2018-06-01'])


# In[84]:


# Define the p, d and q parameters to take any value between 0 and 2
p = range(0,8)
d = range(0,2)
q = range(0,1)


# In[85]:


# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))


# In[86]:


best_aic = np.inf
best_pdq = None
temp_model = None
for param in pdq:   
    temp_model = ARIMA(train,order=param)
    results = temp_model.fit()
    if results.aic < best_aic:
        best_aic = results.aic
        best_pdq = param 
print("Best ARIMA {} model - AIC:{}".format(best_pdq,best_aic))


# ### Ajustamos el modelo

# In[87]:


# using the best parameter in the model
model = ARIMA(train,order=(7,1,0))
model_fit = model.fit()


# In[88]:


model_fit.summary()


# ### Prediccion

# In[89]:


predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)


# In[90]:


# plot results
plt.figure(figsize=(12,6))
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# ### Evaluación del modelo:

# In[91]:


# Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(test, predictions))
rmse = round(rmse, 3)


# In[92]:


# Mean Absolute Percentage Error
abs_error = np.abs(test['demanda_energia']-predictions)
actual = test['demanda_energia']
mape = np.round(np.mean(np.abs(abs_error/actual)),3)


# In[96]:


resultsDf = pd.DataFrame({'Method':['ARIMA'], 'MAPE': [mape], 'RMSE': [rmse]})
resultsDf = resultsDf[['Method', 'RMSE', 'MAPE']]
resultsDf


# El error cuadrático medio (RMSE) es la desviación estándar de los residuos ( errores de predicción ). Los residuos son una medida de qué tan lejos están los puntos de datos de la línea de regresión; RMSE es una medida de la dispersión de estos residuos. En otras palabras, le dice qué tan concentrados están los datos alrededor de la línea de mejor ajuste. 
# 
# 
# El error de porcentaje absoluto medio (MAPE), también conocido como desviación de porcentaje absoluto medio (MAPD), es una medida de precisión de predicción de un método de pronóstico en estadística, por ejemplo, en estimación de tendencias, también utilizado como función de pérdida para problemas de regresión en Machine Learning.

#    

# ### Referencias:
# 
# * _https://towardsdatascience.com/time-series-forecasting-using-auto-arima-in-python-bb83e49210cd_

#    
