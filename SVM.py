import dataPreprocess
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import svm

def main():
    train_X, test_X, train_Y, test_Y =  dataPreprocess.dataprocess()
    #reg_model = Sequential()
    #reg_model.add(Dense(input_dim=33, units=1))      
    #reg_model.compile(loss='mse', optimizer='sgd',metrics=['accuracy'])
    reg_model = svm.SVR()
    reg_model.fit(train_X, train_Y)
    y_test_preds = reg_model.predict(test_X)
    results= r2_score(test_Y, y_test_preds)
    print(results)
    mse = mean_squared_error(test_Y, y_test_preds)
    rmse = mse** 0.5
    print("MSE: %.4f" % mse)
    print("RMSE: %.4f" % rmse)
main()
