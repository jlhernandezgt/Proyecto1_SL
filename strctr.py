# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:39:30 2022

@author: luish
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import fncs_prjct as fn

data = np.load("proyecto_training_data.npy")
data = pd.DataFrame(data)
data.rename(columns={0: "SalePrice", 1: "OverallQual", 2: "1stFlrSF", 3: "TotRmsAbvGrd", 4: "YearBuilt", 5: "LotFrontage"}, inplace = True)

### slicing
training = pd.DataFrame(data[0:1168])
test = pd.DataFrame(data[1169:1460])


###  analisis exploratorio general
columnas_df = fn.select_col_numerics(data)
data_desc = data.describe()
suma_data_null = data.isnull().sum()
mean_data_null = data.isnull().mean()

## histograma variable
for col in columnas_df:
    fn.graficar_data_densidad(training, col) 

for col in columnas_df:
    fn.plot_reg_dens(data, 'SalePrice', col)   

    
###  analisis exploratorio data entrenamiento
tr_desc = training.describe()
tr_sm_null = training.isnull().sum()
tr_mn_null = training.isnull().mean()


  
    
correlacion = training.corr()
sns.heatmap(training.corr(method = "pearson"), annot=True, cmap = "YlGnBu")

fn.plot_regresion_top2(training,'SalePrice','OverallQual','1stFlrSF')


variableX = training['OverallQual']
variableY = training['SalePrice']
variableXtest = test['OverallQual']
variableYtest = test['SalePrice']


xvariable = np.array(variableX)
yvariable = np.array(variableY)
xvtest = np.array(variableXtest)
yvtest = np.array(variableYtest)

"""
a = 2
b = 2
epoch = 1000
learning_rate = 0.01
imprimir_error_cada = 10
"""

fn.training_model(variableX, variableY, 2, 2)


a = 0
b = 0
epoch = 75000
learning_rate = 0.01
imprimir_error_cada = 10

beta0, beta1, resultado_gen, ayb, rmse, estimaciones_g, vr = fn.gradiant_training(vx = xvariable, vy = yvariable, b0 = a, b1 = b, alpha = learning_rate, epochs=epoch)       
 

fn.training_model(xvariable, yvariable , beta0, beta1)  
fn.training_model(xvtest, yvtest, beta0, beta1)     


fn.graf_disper_y_valores_reales(estimaciones_g, vr)

fn.graf_pred_n_epoch(xvariable, yvariable,beta0,beta1, epoch)

fn.fn_plt_error(rmse) 


variableX2 = np.array(variableX).reshape(-1,1)
variableXt2 = np.array(variableXtest).reshape(-1,1)
variableY2 = np.array(variableY).reshape(-1,1)
variableYt2 = np.array(variableYtest).reshape(-1,1)

### validacion de datos con sklearn

reg_ln = LinearRegression()
reg_ln.fit(variableX2,yvariable)
reg_pred = reg_ln.predict(variableXt2)

print("Validacion de Intercepto modelo SKL ", reg_ln.intercept_,
      "VS Validacion de Beta0 Modelo Manual", beta0)
print("Validacion de Coeficiente modelo SKL ", reg_ln.coef_,
      "VS Validacion de Beta1 Modelo Manual", beta1)
print("Validacion de RMSE modelo SKL ", mean_squared_error(variableYt2, reg_pred, squared = False),
      "VS Validacion de RMSE Modelo Manual", rmse[-1])


fn.graf_mdl_vs_skl(variableXt2,yvtest,reg_pred,xvtest, beta0, beta1)


errores = fn.df_errores(xvtest, yvtest, beta0, beta1)

ytest1 = errores['Y']
yestimado1 = errores['Y_ESTIMADO']
gradiante_y = np.asmatrix(ytest1) - np.asmatrix(yestimado1)
skl_y = yvtest - reg_pred

fn.graf_dis_dens_errors_mdls(gradiante_y, skl_y)
