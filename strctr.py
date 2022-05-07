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

entrenamiento = data[0:1168]
pruebas = data[1169:1460]

data = pd.DataFrame(data)
columnas_df = fn.select_col_numerics(data)
data_desc = data.describe()
suma_data_null = data.isnull().sum()
mean_data_null = data.isnull().mean()


for col in columnas_df:
    fn.plot_reg_dens(data, 0, col)   
    

