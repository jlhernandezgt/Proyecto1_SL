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



a = 3
b = 3
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
        ### sumatoria de yesti - y elev al cuadr partido n
        rms = np.sqrt(np.mean((y_estimado - vy)**2))
        rmse.append(rms)     
                
    estimaciones.append(y_estimado)
    valores_reales.append((vy))
    resultados[epochs] = ((b0,b1))
    return b0, b1, resultados, ayb, rmse, estimaciones, valores_reales
            
        
        
        
beta0, beta1, resultado_gen, ayb, rmse, estimaciones_g, vr = gradiant_training(vx = variableX, vy = variableY, b0 = a, b1 = b, alpha = learning_rate, epochs=epoch)       
        
training_model(variableX[0:10], variableY[0:10], beta0, beta1)      



x = estimaciones_g
y = vr        
plt.scatter(x,y, color = '#88c999')
plt.xlabel("Estimaciones")
plt.ylabel("Valores reales")
plt.title("Dispersión entre las estimaciones y valores reales")      



plt.scatter(variableX, variableY)
pred_x = [1, max(variableX)]
pred_y = [beta0 + beta1*0, beta0+beta1*max(variableX)]
plt.plot(pred_x, pred_y, 'r')
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.title('Prediccion de ventas con ' + str(epoch) + " iteraciones")
plt.show()


 def fn_plt_error (df):
    plt.figure(figsize = (16,3))
    plt.plot(df, 'm--')
    plt.title('Evolución Error Mínimo', fontSize = 16)
    plt.xlabel('Iteracion')
    plt.ylabel('RMSE')
    plt.annotate('Inicio Regulacion Error Miimo', xy=(200, 50000), xytext=(4000, 50000),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )
    plt.show()  
        
fn_plt_error(rmse) 



from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


sk_training = pd.DataFrame(data[0:1168])
sk_test = pd.DataFrame(data[1169:1460])
sk_variableX = training['OverallQual']
sk_variableY = training['SalePrice']

sk_variableX = np.array(sk_variableX).reshape(-1,1)
sk_variableY = np.array(sk_variableY).reshape(-1,1)


rl = LinearRegression()
rl.fit(sk_variableX, sk_variableY)

prueba = rl.predict(sk_variableX)

print("Validacion de Intercepto modelo SKL ", rl.intercept_,
      "VS Validacion de Beta0 Modelo Manual", beta0)
print("Validacion de Coeficiente modelo SKL ", rl.coef_,
      "VS Validacion de Beta1 Modelo Manual", beta1)
print("Validacion de RMSE modelo SKL ", mean_squared_error(sk_variableY, prueba, squared = False),
      "VS Validacion de RMSE Modelo Manual", rmse[-1])


