# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:39:30 2022

@author: luish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fncs_prjct as fn

data = np.load("proyecto_training_data.npy")
data = pd.DataFrame(data)

### slicing
training = pd.DataFrame(data[0:1168])
test = pd.DataFrame(data[1169:1460])


###  analisis exploratorio general
columnas_df = fn.select_col_numerics(data)
data_desc = data.describe()
suma_data_null = data.isnull().sum()
mean_data_null = data.isnull().mean()

for col in columnas_df:
    fn.plot_reg_dens(data, 0, col)   

    
###  analisis exploratorio data entrenamiento
tr_desc = training.describe()
tr_sm_null = training.isnull().sum()
tr_mn_null = training.isnull().mean()


## histograma variable
for col in columnas_df:
    fn.graficar_data_densidad(training, col)  

    
