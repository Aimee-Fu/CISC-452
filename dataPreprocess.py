import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA

def dataprocess():
    
    data = pd. read_csv("//Users/tsukeka/Downloads/LengthOfStay.csv")
    data.drop(columns = ["eid","vdate","discharged","facid"],inplace = True)
    data = data. replace({'gender': {'M':1, 'F':0}, 'rcount': {'5+' : 5}})
    data = data.astype({'rcount': 'int'})
    
    hematocrit = data[['hematocrit']].values
    data['hematocrit'] = preprocessing.StandardScaler().fit_transform(hematocrit)
    
    bloodureanitro = data[['neutrophils']].values
    data['neutrophils'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    sodium = data[['sodium']].values
    data['sodium'] = preprocessing.StandardScaler().fit_transform(sodium)

    glucose = data[['glucose']].values
    data['glucose'] = preprocessing.StandardScaler().fit_transform(glucose)

    bloodureanitro = data[['bloodureanitro']].values
    data['bloodureanitro'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    creatinine = data[['creatinine']].values
    data['creatinine'] = preprocessing.StandardScaler().fit_transform(creatinine)

    bmi = data[['bmi']].values
    data['bmi'] = preprocessing.StandardScaler().fit_transform(bmi)

    pulse = data[['pulse']].values
    data['pulse'] = preprocessing.StandardScaler().fit_transform(pulse)

    respiration = data[['respiration']].values
    data['respiration'] = preprocessing.StandardScaler().fit_transform(respiration)

    data = pd.concat([data, pd.get_dummies(data['secondarydiagnosisnonicd9'])], axis=1)
    data = data.drop(columns=['secondarydiagnosisnonicd9'])
    labels = data['lengthofstay']
    data = data.drop(columns=['lengthofstay'])

    pca = PCA()   
    data=pca.fit_transform(data)
    
    train_X = np.array(data[:80000])
    train_Y = labels.head(n=80000).to_numpy()
    test_X = np.array(data[80000:])
    test_Y = labels.tail(n=20000).to_numpy()
    return train_X, test_X, train_Y, test_Y
