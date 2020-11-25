import pandas as pd
import numpy as np
import dataPreprocess
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras.layers import Dropout
import numpy as np

def annModel(train_X, test_X, train_Y, test_Y):
    model = Sequential()
    model.add(Dense(units = 66,  input_dim =33 ,  kernel_initializer =  'normal' ,  activation = 'sigmoid'))
    model.add(Dense(units = 16, kernel_initializer =  'normal',activation = 'relu'))
    model.add(Dense(units = 1, kernel_initializer =  'normal'))
    adam=optimizers.Adam(lr=0.001,  epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_squared_error') 
    model.fit(train_X, train_Y, epochs=200,verbose=2)
    loss = model.evaluate(test_X,  test_Y,verbose=2)
    print(loss)
def main():
     train_X, test_X, train_Y, test_Y=  dataPreprocess.dataprocess()
     annModel(train_X, test_X, train_Y, test_Y)
main()
     
