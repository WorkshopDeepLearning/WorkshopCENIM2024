# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:38:23 2024

@author: Sevito Fernanda P
"""

#Save this program with the name "ModulDeepLearningMLP.py"
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model

def Training(X,T, JumEpoh,NamaFileBobot):
    #==================================================
    #Step 2: defining the JST model
    model = Sequential()

    NInput = X.shape[1] # Number of input vectors
    NNOutput = T.shape[1] # Number of Nodes.
    #Adding a Dense Layer to the Model
    model.add(Dense(1000, input_dim=NInput, activation='relu',use_bias=True))
    model.add(Dense(1000,  activation='relu',use_bias=True))
    model.add(Dense(1000,  activation='relu',use_bias=True))
    model.add(Dense(NNOutput,  activation='sigmoid',use_bias=True))
    model.compile(loss='mse',optimizer='SGD',metrics=['accuracy'])
    His=model.fit(X, T,epochs=JumEpoh)
    plt.plot(His.history['loss'])
    print(model.summary())
    model.save(NamaFileBobot)
    return model,His

def Prediction(X):
    model=load_model(NamaFileBobot)
    yp=model.predict(X)
    return yp