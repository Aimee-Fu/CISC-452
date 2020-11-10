import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def read():
    data = pd. read_csv("https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv")
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

    n1=np.mean(data['pulse'])
    print(n1)
    plt.hist(data['pulse'], bins=100)
    plt.show()
    #print(data['neutrophils'],"\n")

    #sodium = data[['sodium']].values
    #data['sodium'] = preprocessing.StandardScaler().fit_transform(sodium)
read()
