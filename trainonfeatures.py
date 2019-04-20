# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:40:51 2019

@author: Shalin
"""
import sys
import argparse
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', action='store', type=int, dest="type",\
                        help="model to be used = 0:SVM,1:RandomForestClassifier,2:LogisticClassifier, 3:Naive Bayes",choices=range(0,4), required=True)

    return parser.parse_args(sys.argv[1:])

def featureExtraction(x, fs):
    f1 = np.mean(x,0).reshape((1,3))[0] # Means
    C = np.cov(x.T)
    f2=[]
    f2 = np.append(C[0][0:3],C[1][1:3])
    f2 = np.array(f2,dtype=np.float32)
    f2 = np.append(f2,[C[2][2]])
    f3 = (skew(x[:,0]), skew(x[:,1]), skew(x[:,2])) # Skewness
    f4 = (kurtosis(x[:,0], fisher=False), kurtosis(x[:,1], fisher=False), kurtosis(x[:,2],fisher=False)) # Kurtosis
    f5 = np.zeros(3)
    f6 = np.zeros(3)
    F = []
    for i in range(0,3):
        g = abs(np.fft.fft(x[:,i]))
        g = g[0:round(len(g)/2)]
        g[0] = 0
        w = fs * np.arange(0,len(g))/(2*len(g))
        v = max(g)
        idx = np.argmax(g)
        f5[i] = v
        f6[i] = w[idx]

# Putting together feature vector
    F = np.append(f1, f2)
    F = np.append(F,f3)
    F = np.append(F,f4)
    F = np.append(F,f5)
    F = np.append(F,f6)
    return F

def output(y,batch_size):
    size = int(np.ceil(len(y)/batch_size))
    y_new=[]
    for i in range(0,size):
        count_zero = (y[i*batch_size:i*batch_size+batch_size]==0).sum()
        count_one = (y[i*batch_size:i*batch_size+batch_size]==1).sum()
        if count_zero >= count_one:
            y_new.append(0)
        else:
            y_new.append(1)
    return y_new

def model(type=1,n_estimators=10):
    if type==0:
        from sklearn.svm import SVC
        print('SVM model')
        classifier = SVC(kernel='rbf', random_state = 0)
    elif type==1:
        from sklearn.ensemble import RandomForestClassifier
        print('RandomForest model')
        classifier = RandomForestClassifier(max_depth = 80,max_features = 2, min_samples_leaf = 4,min_samples_split = 8,n_estimators=300,random_state = 0)
    elif type==2:
        from sklearn.linear_model import LogisticRegression
        print('Linear model')
        classifier = LogisticRegression(random_state=0)
    elif type==3:
        from sklearn.naive_bayes import GaussianNB
        print('Naive Bayes model')
        classifier = GaussianNB()
        
    return classifier

def fit_model(classifier,X,y):
    return classifier.fit(X,y)

def predict(classifier,X,y):
    #print score
    print(classifier.score(X, y))
    
    #predict
    y_pred= classifier.predict(X)

    cm=confusion_matrix(y,y_pred)
    return y_pred, cm

def features(arm,wrist,fs=50,batch_size=50):
    X=[]
    for i in range(0,len(wrist)-batch_size):
#        print(i)
        X.append(featureExtraction(arm[i:i+batch_size,:3],fs).tolist())
        X[i].extend(featureExtraction(arm[i:i+batch_size,3:],fs).tolist())
        X[i].extend(featureExtraction(wrist[i:i+batch_size,:3],fs).tolist())
        X[i].extend(featureExtraction(wrist[i:i+batch_size,3:],fs).tolist())  
    return X

if __name__ == '__main__':
    start_time = datetime.now()
    FLAGS = get_args()
    fs = 50
    batch_size = 150
    print('Training')
    X_train = pd.read_csv('Train_Features/train_features.csv', header=None)#, dtype=np.float32)
    y_train = pd.read_csv('Train_Features/train_labels.csv', header=None)#, dtype=np.float32)
    X_train = X_train.as_matrix()
    y_train = y_train.as_matrix()
    #create classifier
    print('define classifier')
    if FLAGS.type == 1:
        n_estimators = 50
        classifier = model(n_estimators=n_estimators,type=FLAGS.type)
    else:
        classifier = model(type=FLAGS.type)
    
    print('fit model')
    classifier = fit_model(classifier,X_train,y_train)

    test_session=['Session02','Session03','Session15','Session16']
    X_test={}
    y_pred={}
    for i in test_session:   
        print(i)
        X_test[i] = pd.read_csv('Test_Features/'+i+'/features.csv', delim_whitespace= True, skipinitialspace= True, header=None)
        X_test[i] = X_test.as_matrix()
        
        #predict
        y_pred[i] = predict(classifier, X_test[i])
        y_pred[i] = np.append([0]*batch_size,y_pred[i])
        df = pd.DataFrame(y_pred[i])
        df.to_csv('Test_Features/'+i+'/prediction.txt', index=False, header=False)   
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))