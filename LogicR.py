import dataPreprocess
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
import warnings

def main():
    train_X, test_X, train_Y, test_Y =  dataPreprocess.dataprocess()
    #reg_model  = LR()
    reg_model  = LogisticRegression(max_iter=3000)
    reg_model.fit(train_X, train_Y)
    y_test_preds = reg_model.predict(test_X)
    results= r2_score(test_Y, y_test_preds)
    print(results)
    mse = mean_squared_error(test_Y, y_test_preds)
    rmse = mse** 0.5
    print("MSE: %.4f" % mse)
    print("RMSE: %.4f" % rmse)
    print("Testing score:%f"%reg_model.score(test_X,test_Y ))

main()
