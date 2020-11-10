import dataPreprocess
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


def main():
    train_X, test_X, train_Y, test_Y =  dataPreprocess.dataprocess()
    reg_model = GradientBoostingRegressor(random_state = 0)
    reg_model.fit(train_X, train_Y)
    y_test_preds = reg_model.predict(test_X)
    results= r2_score(test_Y, y_test_preds)
    print(results)
    mse = mean_squared_error(test_Y, y_test_preds)
    print("MSE: %.4f" % mse)
main()
