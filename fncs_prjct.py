# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:47:45 2022

@author: luish
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



def graficar_data_densidad(df, col1):
    sns.set_style("dark")
    x = df[col1]
    ax = sns.distplot(x, color ='#20DF97')
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
    sns.set_style("dark")
    x = df[col1]
    ax = sns.distplot(x, color ='#20DF97')
    plt.title("Densidad-Histograma: " + str(col1))
    
    plt.subplot(122)
    sns.set_style("dark")
    sns.regplot(x=col2, y=col1, data=df,color="#BE9ED3");
    plt.title("Regresion : " + str(col2))
    plt.show()


    

def plot_regresion_top2(df, col1, col2, col3):
    correlacion = df.corr()
    a = round(correlacion.iloc[0,1],5)
    b = round(correlacion.iloc[0,2],5)
    
    plt.figure(figsize = (16,6))
    
    fig, ax = plt.subplots()
    varx = df[col2]
    vary = df[col1]
    sns.set_style("dark")
    sns.regplot(x=varx, y=vary, data=df,color="#20DF97");
    plt.title("Coeficiente de Correlacion OverallQual y SalePrice: " + str(a))
    plt.show()

    fig, ax = plt.subplots()
    varx = df[col3]
    vary = df[col1]
    sns.set_style("dark")
    sns.regplot(x=varx, y=vary, data=df,color="#BE9ED3");
    plt.title("Coeficiente de Correlacion 1stFlrSF y SalePrice: " + str(b))
    plt.show()
    



def training_model(vx, vy, b0, b1):
    n = len(vx)
    resultados = {}
    for i in range(n):
        y_estimado = b0+b1*vx[i]
        error_estimado = (vy[i] - y_estimado) 
        vc = 1 / (2*len(vx))
        error = vc * error_estimado**2
        evento = i
        resultados[evento] = vx[i], vy[i], y_estimado, error_estimado, error, b0, b1
    return(resultados)




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
    imprimir_error_cada = 10
    
    for i in range(epochs):
        y_estimado = b0+b1*vx
        g_a = 1/n * np.sum((y_estimado - vy))
        g_b = 1/n * np.sum((y_estimado - vy)*vx)
        b0 = b0-alpha*g_a
        b1 = b1-alpha*g_b
        g_a = np.append(g_a, [g_a])
        g_b = np.append(g_b, [g_b])
        ayb.append((b0, b1))
        ### sumatoria de yesti - y elev al cuadr partido n
        if((i % imprimir_error_cada)==0):
            rms = np.sqrt(np.mean((y_estimado - vy)**2))
        rmse.append(rms)    
                
    estimaciones.append(y_estimado)
    valores_reales.append((vy))
    resultados[epochs] = ((b0,b1))
    return b0, b1, resultados, ayb, rmse, estimaciones, valores_reales
            

        
        
def graf_disper_y_valores_reales(x,y):          
    plt.scatter(x,y, color = '#20DF97')
    plt.xlabel("Estimaciones")
    plt.ylabel("Valores reales")
    plt.title("Dispersión entre las estimaciones y valores reales")      




def graf_pred_n_epoch(x, y, b0, b1, epochs):
    plt.scatter(x, y,color = '#20DF97')
    pred_x = [1, max(x)]
    pred_y = [b0 + b1*0, b0+b1*max(x)]
    plt.plot(pred_x, pred_y, 'r')
    plt.xlabel('OverallQual')
    plt.ylabel('SalePrice')
    plt.title('Prediccion de ventas con ' + str(epochs) + " iteraciones")
    plt.show()




def fn_plt_error (df):
    plt.figure(figsize = (16,6))
    plt.plot(df, 'm--')
    plt.title('Evolución Error Mínimo', fontSize = 16)
    plt.xlabel('Iteracion')
    plt.ylabel('RMSE')
    plt.annotate('Inicio Regulacion Error Miimo', xy=(250, 50000), xytext=(4000, 50000),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )
    plt.show()  
        



def graf_mdl_vs_skl(xt, yt, pred, vx, b0, b1):
    plt.scatter(xt, yt, color = '#20DF97')
    plt.plot(xt, pred, color = 'red', linestyle = 'dashed')
    pred_x = [1, max(vx)]
    pred_y = [b0 + b1*0, b0+b1*max(vx)]
    plt.plot(pred_x, pred_y, color = 'blue', linestyle = 'dashdot')




def df_errores(xt, yt, b0, b1):
    df = training_model(xt, yt, b0, b1)        
    df = pd.DataFrame.from_dict(df, orient = 'index')
    df.rename(columns={0:"X", 1:"Y", 2:"Y_ESTIMADO", 3:"ERROR", 4:"ERROR_CUADRATICO", 5:"BETA0", 6:"BETA1"}, inplace = True)
    return(df)




def graf_dis_dens_errors_mdls(gy_mdl,  gy_skl):
    sns.set_style("dark")
    sns.kdeplot(np.asarray(gy_mdl)[0], label = 'Error Modelo Manual', color = '#BE9ED3')
    sns.kdeplot(np.array(gy_skl), label = 'Error SKL', color = '#20DF97', linestyle =  '--')
    plt.title('Distribucion de Error Ambos Modelos')
    plt.xlabel("Error")
    plt.ylabel('Distribucion Densidad')
    plt.legend()
    plt.show()










