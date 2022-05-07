# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:39:30 2022

@author: luish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load("proyecto_training_data.npy")
data = pd.DataFrame(data)

def getContinuesCols(df):
    colnames = df.columns
    numeric_continues_vars = []
    for col in colnames:
        unique_values =len (df[col].unique())
        if((df[col].dtype != 'object') and (unique_values > 0)):
            numeric_continues_vars.append(col)
    return numeric_continues_vars

analizar = getContinuesCols(data)

def plot_density(df, variable):
    sns.set_theme(); 
    x = df[variable]
    ax = sns.distplot(x)
    
for col in analizar:
        plot_density(data, col)

data[0].describe()
data[1].describe()
data[2].describe()
data[3].describe()
data[4].describe()
data[5].describe()



data[0].isnull().sum()
data[1].isnull().sum()
data[2].isnull().sum()
data[3].isnull().sum()
data[4].isnull().sum()

data[5].isnull().sum()
data[5].isnull().mean()



sns.set_theme(); 
x = data[0]
ax = sns.distplot(x)

sns.set_theme(); 
x = data[1]
ax = sns.distplot(x)

sns.set_theme(); 
x = data[2]
ax = sns.distplot(x)

sns.set_theme(); 
x = data[3]
ax = sns.distplot(x)

sns.set_theme(); 
x = data[4]
ax = sns.distplot(x)

sns.set_theme(); 
x = data[5]
ax = sns.distplot(x)


