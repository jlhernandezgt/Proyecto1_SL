# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:47:45 2022

@author: luish
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def graficar_data_densidad(df, col1):
    sns.set_theme();
    x = df[col1]
    ax = sns.distplot(x)
    plt.title("Densidad-Histograma: " + str(col1))
    plt.show()



def select_col_numerics(df):
    colnames = df.columns
    numeric_continues_vars = []
    for col in colnames:
        unique_values =len (df[col].unique())
        if((df[col].dtype != 'object') and (unique_values > 0)):
            numeric_continues_vars.append(col)
    return numeric_continues_vars    




def plot_reg_dens(df, col1, col2):    
    plt.figure(figsize = (15,6))
    
    plt.subplot(121)
    sns.set_theme();
    x = df[col1]
    ax = sns.distplot(x)
    plt.title("Densidad-Histograma: " + str(col1))
    
    plt.subplot(122)
    fig, ax = plt.subplots()
    varx = df[col2]
    vary = df[col1]
    ax.scatter(varx, vary)
    z = np.polyfit(varx, vary, 1)
    p = np.poly1d(z)
    plt.plot(varx,p(varx), "r--")
    plt.title("Regresion : " + str(col1))
    plt.show()


def plot_regresion_top2(df, col1, col2, col3):
    correlacion = df.corr()
    a = round(correlacion.iloc[0,1],5)
    b = round(correlacion.iloc[0,2],5)
    
    plt.figure(figsize = (16,6))
    
    #plt.subplot(121)
    fig, ax = plt.subplots()
    varx = df[col2]
    vary = df[col1]
    ax.scatter(varx, vary)
    z = np.polyfit(varx, vary, 1)
    p = np.poly1d(z)
    plt.plot(varx,p(varx), "r--")
    plt.title("Coeficiente de Correlacion 1 y 0: " + str(a))
    plt.show()

    #plt.subplot(122)
    fig, ax = plt.subplots()
    varx = df[col3]
    vary = df[col1]
    ax.scatter(varx, vary)
    z = np.polyfit(varx, vary, 1)
    p = np.poly1d(z)
    plt.plot(varx,p(varx), "r--")
    plt.title("Coeficiente de Correlacion 2 y 0: " + str(b))
    plt.show()