#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 19:25:35 2022

@author: andres.garcia.medina@uabc.edu.mx
"""



#######################################
# Libraries
######################################
import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import functions as functions_lstm
import random
import json
import re
import sys


#######################################
### Set enviroment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = os.getcwd()
path_parent = os.path.dirname(os.getcwd())
random.seed(800)


#######################################
### Parallel parameters
case = str(sys.argv[1]) 
k = int(sys.argv[2]) 


#######################################
### Configuration
twind = 21 # days
nseq   = 72
npred  = 24
nval  = 48
ntest = 1
n_repeats = 10
print(k, twind,nseq,npred,nval, ntest, n_repeats)

#######################################
### load and preprocessing data
vec = pd.read_csv(path_parent+"/datos/varianzas_intrahora.csv", index_col=0)

if case=='LSTM' or case=='MLP':
    print(case)
    # choose data window
    datak = np.array(vec.iloc[k*npred:((twind+k)*npred + npred), :])
    # scaling
    scaler = MinMaxScaler(feature_range = (0,1))
    data = scaler.fit_transform(datak)
    
elif case=='LSTM_Vol':
    print(case)
    #load volumen
    vol_ew = pd.read_csv(path_parent+"/datos/volumen_intrahora.csv", index_col=0)
    #concatenate features
    vec_vol = pd.concat((vec, vol_ew), axis=1)
    vec = vec_vol
    # choose data window
    datak = np.array(vec.iloc[k*npred:((twind+k)*npred + npred), :])
    # scaling
    scaler = MinMaxScaler(feature_range = (0,1))
    data = scaler.fit_transform(datak)
    
elif case=='LSTM_eGARCH' or case=='LSTM_gjrGARCH':
    print(case)
    suffix = re.findall('(?<=_).*$',case)
    # choose data window
    datak = np.array(vec.iloc[k*npred:((twind+k)*npred + npred), :])
    # scaling
    scaler = MinMaxScaler(feature_range = (0,1))
    datak_s = scaler.fit_transform(datak)
    # load parameters and choose window
    parametros = pd.read_csv(path_parent+"/datos/parametros_univariado_"+str(suffix[0])+"_sinvol.csv", index_col=0).iloc[:,k]
    parametros_repeat = np.repeat(np.transpose(np.array(parametros).reshape(-1, 1)),(twind+1)*npred,axis=0)
    # stack features
    data = np.column_stack((datak_s,parametros_repeat))




#######################################
### Transform data
#######################################
# supervised data
X,y = functions_lstm.to_supervised(data, nseq, npred)
ntrain  = X.shape[0]-ntest-nval


# train, validation and test datasets
TRAIN_x,TRAIN_y = X[:ntrain,:,:],y[:ntrain,:]
VAL_x,VAL_y = X[ntrain:(ntrain+nval),:,:],y[ntrain:(ntrain+nval),:]
TEST_x,TEST_y = X[-ntest:,:,:],y[-ntest:,:]


print('verify dimensions of datasets')
print('train:',TRAIN_x.shape, TRAIN_y.shape)
print('valid:',VAL_x.shape, VAL_y.shape)
print('test:',TEST_x.shape, TEST_y.shape)

#######################################
### validation and test
#######################################


# model configs
cfg_list = functions_lstm.model_configs()

# grid search
start = time.time()
scores = functions_lstm.grid_search_evaluate(TRAIN_x,TRAIN_y, VAL_x, VAL_y, \
                      TEST_x, TEST_y,nseq,npred,n_repeats,cfg_list,k,case)

done = time.time()
elapsed_total = done - start
print('tiempo de computo final: ',elapsed_total)

#######################################
### save scores
#######################################

# list top configs in descending order
i=0

config_grid = list()
m_mse_grid = list()
s_mse_grid = list()

m_mse_test_grid = list()
s_mse_test_grid = list()

m_predicted_grid = list()
s_predicted_grid = list()

elapsed_grid = list()

for config, m_mse, s_mse, m_mse_test, s_mse_test, m_predicted, s_predicted, elapsed in scores:
    print(i, config, m_mse, s_mse, m_mse_test, s_mse_test, m_predicted, s_predicted, elapsed)
    
    config_grid.append(config)
    
    #validation
    m_mse_grid.append(m_mse)
    s_mse_grid.append(s_mse)

    #test
    m_mse_test_grid.append(m_mse_test)
    s_mse_test_grid.append(s_mse_test)

    #predicted
    m_predicted_grid.append(m_predicted)
    s_predicted_grid.append(s_predicted)
    i+=1

# format to predicted values
m_predicted_lists = [l.tolist() for l in m_predicted_grid]
s_predicted_lists = [l.tolist() for l in s_predicted_grid]

#save all 
results = {'parameters': 
           {'k':k, 'twind':twind, 'nseq':nseq, 'npred':npred, 'nval':nval, 'ntest':ntest, 'n_repeats':n_repeats},
           'elapsed_total': elapsed_total,
           'config_grid':config_grid, 
           'm_mse_grid':m_mse_grid,
           's_mse_grid':s_mse_grid,
           'm_mse_test':m_mse_test_grid,
           's_mse_test_grid':s_mse_test_grid,
           'm_predicted_lists':m_predicted_lists,
           's_predicted_lists':s_predicted_lists}
out_file = open(path_parent+"/results/"+case+"/output_"+str(k)+".json", "w") 
json.dump(results, out_file, indent = 6) 
out_file.close() 


### inverse scaling, save, and plot predictions
if case=='LSTM_Vol':
    real_scaled = TEST_y[0].reshape(-1, 1)
    features_scaled = data[-npred:,1].reshape(-1, 1)
    forecast_scaled = m_predicted_grid[0].reshape(-1, 1)
    real = scaler.inverse_transform(np.concatenate((real_scaled,features_scaled),axis=1))[:,0]
    forecast = scaler.inverse_transform(np.concatenate((forecast_scaled,features_scaled),axis=1))[:,0]

else:
    real_scaled = TEST_y[0].reshape(-1, 1)
    forecast_scaled = m_predicted_grid[0].reshape(-1, 1)
    real = scaler.inverse_transform(real_scaled)
    forecast = scaler.inverse_transform(forecast_scaled)


predicciones = {'real': real.ravel(), 'forecast':forecast.ravel()}
df_predicciones = pd.DataFrame(predicciones)
export_csv = df_predicciones.to_csv (path_parent+"/results/"+case+"/predictions_"+str(k)+".csv", index = None, header=True)


plt.figure()
plt.plot(real.ravel(),label='real')
plt.plot(forecast.ravel(),label='forecast')
plt.legend()
plt.savefig(path_parent+"/figures/"+case+"/forecast_"+str(k)+".png",dpi=600,bbox_inches='tight')
#plt.show()

