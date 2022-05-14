# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:47:45 2022

@author: luish
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

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
    fig, ay = plt.subplots()
    varx = df[col2]
    vary = df[col1]
    ay.scatter(varx, vary)
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
    



variableX = training['OverallQual']
variableY = training['SalePrice']
sumaX = variableX.sum()
sumaY = variableY.sum()
promedioX = variableX.mean()
promedioY = variableY.mean()
xy = variableX*variableY
x2 = variableX**2
numerador = xy.sum()-len(variableX)*promedioX*promedioY
denominador = x2.sum()-len(variableX)*promedioX**2
beta1 = numerador/denominador
beta0 = promedioY-beta1*promedioX


a = 2
b = 2
epoch = 1000
learning_rate = 0.01
imprimir_error_cada = 10


def training_model(vx, vy, b0, b1):
    n = len(vx)
    resultados = {}
    for i in range(n):
        y_estimado = b0+b1*vx[i]
        error_estimado = (vy[i] - y_estimado) 
        vc = 1 / (2*len(variableX))
        error = vc * error_estimado**2
        evento = i
        resultados[evento] = vx[i], vy[i], y_estimado, error_estimado, error, b0, b1
    return(resultados)



a = 2
b = 2
epoch = 75000
learning_rate = 0.01
imprimir_error_cada = 10

def gradiant_training(vx, vy, b0, b1, alpha, epochs):
    n = len(vx)
    g_a = np.array([])
    g_b = np.array([])
    g_f = np.column_stack((g_a,g_b))
    ayb = []
    rmse = []
    valores_reales = []
    estimaciones = []
    resultados = {}
    
    for i in range(epochs):
        y_estimado = b0+b1*vx
        g_a = 1/n * np.sum((y_estimado - vy))
        g_b = 1/n * np.sum((y_estimado - vy)*vx)
        b0 = b0-alpha*g_a
        b1 = b1-alpha*g_b
        g_a = np.append(g_a, [g_a])
        g_b = np.append(g_b, [g_b])
        ayb.append((b0, b1))
        #rmse.append(mean_squared_error(vy, y_estimado, squared = False))
    estimaciones.append(y_estimado)
    valores_reales.append((vy))
    resultados[epochs] = ((b0,b1))
    return b0, b1, resultados, ayb, estimaciones, valores_reales
            
        
        
        
beta0, beta1, resultado_gen, ayb, estimaciones_g, vr = gradiant_training(vx = variableX, vy = variableY, b0 = a, b1 = b, alpha = learning_rate, epochs=epoch)       
        
        

b0  -96469.57131873941      
b1  45411.99877915908
        
        
        
        



