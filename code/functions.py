#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:17:40 2022

@author: andres
"""


#######################################
### Libraries
#######################################
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
import os

path = os.getcwd()
path_parent = os.path.dirname(os.getcwd())
#######################################
### Functions
#######################################


# convert history into inputs and outputs
def to_supervised(data, n_input, n_out):
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)

# create a list of configs to try 
def model_configs():
    # define scope of configs
    funac_act = ["sigmoid"]
    batch_sr = [24, 48, 72]
    # create configs
    configs = list()
    for i in funac_act:
        for j in batch_sr:
            cfg = [i, j]
            configs.append(cfg)
    print('Total configs: %d' % len(configs))
    return configs

# grid search configs
def grid_search_evaluate(train_x,train_y, val_x, val_y, test_x, test_y, n_input,n_output,n_repeats,cfg_list,k,case):
    # evaluate configs
    scores = [run_experiment(train_x,train_y, val_x, val_y, test_x, test_y,n_input,n_output,n_repeats,cfg,k,case) for cfg in cfg_list]
    # sort configs by error, asc
    scores.sort(reverse=False, key=lambda tup: tup[1])
    return(scores)

def run_experiment(train_x,train_y, val_x, val_y, test_x, test_y,n_input,n_output,n_repeats,cfg,k,case):
    config = str(cfg)
    # load data
    funac_act= cfg[0]
    batch_sr = cfg[1]

    # repeat experiment
    MSE = list()
    MSE_test = list()
    predicted = list()
    
    start = time.time()
    for r in range(n_repeats):
        mse,mse_test, summary, best_model  = \
            evaluate_model(batch_sr, funac_act, train_x, train_y, val_x, val_y, test_x,test_y,k,case)
            
        #validation
        MSE.append(mse)
        
        #plot learning curves
        if r==0:
            plt.figure()
            plt.plot(summary.history['loss'], label='loss train')
            plt.plot(summary.history['val_loss'], label='loss validation')
            plt.xlabel('epochs')
            plt.legend()
            plt.savefig(path_parent+"/figures/"+case+"/learning_curve_k_"+str(k)+"_batch_"+str(batch_sr)+".png",dpi=600,bbox_inches='tight')

        
        #test
        MSE_test.append(mse_test)

        #predict
        predicty = best_model.predict(test_x)
        predicted.append(predicty.ravel())
    
    done = time.time()
    elapsed = done - start
        
    # summarize results: validation
    m_mse, s_mse = np.mean(MSE), np.std(MSE)

        
    # summarize results: test
    m_mse_test, s_mse_test = np.mean(MSE_test), np.std(MSE_test)
    
    
    # summarize results: predicted
    m_predicted, s_predicted = np.mean(predicted,axis=0),np.std(predicted,axis=0) 
    
    print('config: ', config,'tiempo de computo: ',elapsed) 
    return(config, m_mse,s_mse, m_mse_test,s_mse_test, m_predicted, s_predicted, elapsed)


# fit and evaluate a model
def evaluate_model (batch, funac, train_x, train_y, val_x, val_y, test_x, test_y,k,case):
    _, n_timesteps, n_features, npred = train_x.shape[0], train_x.shape[1],train_x.shape[2], train_y.shape[1]  
    epochs = 100
    print('#features:',n_features)
    

    if case=='MLP':
        model = Sequential()
        model.add(Dense(10, activation=funac, input_dim=n_timesteps))
        model.add(Dense(npred))

    else:
        model = Sequential()
        model.add(LSTM(10, input_shape=(n_timesteps,n_features), return_sequences=True, activation=funac))
        model.add(Dropout(0.3))
        model.add(LSTM(4, return_sequences=True, activation=funac))
        model.add(Dropout(0.4))
        model.add(LSTM(2, return_sequences=False, activation=funac))
        #model.add(Dropout(0.4))
        model.add(Dense(5))
        model.add(Dense(npred, activation=funac))
    
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mae', optimizer=opt)
        
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
    mc = ModelCheckpoint(path_parent+'/results/'+case+'/best_model_'+str(k)+'.h5',monitor='val_loss',mode='min',\
                         verbose=0, save_best_only=True)
        
    # fit network
    history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=epochs,\
                        batch_size=batch, verbose=0,  callbacks=[es, mc])
        
    # load the saved model
    best_model = load_model(path_parent+'/results/'+case+'/best_model_'+str(k)+'.h5')
    
    # evaluate model
    mse = best_model.evaluate(val_x, val_y, batch_size=batch, verbose=0)
    mse_test = best_model.evaluate(test_x, test_y, batch_size=batch, verbose=0)
    model.reset_states()
    return  mse,mse_test, history, best_model
    
