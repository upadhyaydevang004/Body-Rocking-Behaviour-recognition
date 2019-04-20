# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:38:58 2019

@author: Shalin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:40:51 2019

@author: Shalin
"""
import numpy as np
from scipy.stats import kurtosis, skew
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix

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
        classifier = RandomForestClassifier(random_state = 0)
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
    return classifier.fit(X,np.ravel(y))

def predict(classifier,X):
    #print score
#    print(classifier.score(X, y))
    
    #predict
    y_pred= classifier.predict(X)

#    cm=confusion_matrix(y,y_pred)
    return y_pred

def features(arm,wrist,fs=50,batch_size=50):
    X=[]
    for i in range(0,len(wrist)-batch_size):
#        print(i)
        X.append(featureExtraction(arm[i:i+batch_size,:3],fs).tolist())
# =============================================================================
#         X[i] = np.append(X[i],featureExtraction(arm[i:i+batch_size,3:],fs))
#         X[i] = np.append(X[i],featureExtraction(wrist[i:i+batch_size,:3],fs))
#         X[i] = np.append(X[i],featureExtraction(wrist[i:i+batch_size,3:],fs))    
# =============================================================================
        X[i].extend(featureExtraction(arm[i:i+batch_size,3:],fs).tolist())
        X[i].extend(featureExtraction(wrist[i:i+batch_size,:3],fs).tolist())
        X[i].extend(featureExtraction(wrist[i:i+batch_size,3:],fs).tolist())  
    return X

if __name__ == '__main__':
    fs = 50
    batch_size = 150
    train_session = ['Session13','Session01','Session05','Session06','Session07','Session12']
    test_session = ['Session02','Session03','Session15','Session16']
    X_train=[]
    y_train=[]
    print('Training')
    for i in train_session:
        print(i[:-2]+' '+i[-2:])
        arm_train = pd.read_csv('Training Data/'+i+'/armIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        wrist_train = pd.read_csv('Training Data/'+i+'/wristIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        y = pd.read_csv('Training Data/'+i+'/detection.txt',skipinitialspace=True,header=None)
        y = y.as_matrix()
    
        arm_train = arm_train.as_matrix()
        wrist_train = wrist_train.as_matrix()
#        X_train.append(features(fs=fs,batch_size=batch_size,arm=arm_train, wrist=wrist_train))
        X = features(fs=fs,batch_size=batch_size,arm=arm_train, wrist=wrist_train)
        for i in X:
            X_train.append(i)
        #size_train = int(np.ceil(len(arm_train)/batch_size))
        y_train = np.append(y_train,y[batch_size:])
    
    #feature scaling
    print('feature scaling')
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    df = pd.DataFrame(X_train)
    df.to_csv('Train_Features/train_features.csv', index=False, header=False) 
    df = pd.DataFrame(y_train)
    df.to_csv('Train_features/train_labels.csv',index=False,header=False)
    
    for i in test_session:
        print(i[:-2]+' '+i[-2:])
        arm_train = pd.read_csv('Test Data 1/'+i+'/armIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
        wrist_train = pd.read_csv('Test Data 1/'+i+'/wristIMU.txt', delim_whitespace= True, skipinitialspace= True, header=None)
    
        arm_train = arm_train.as_matrix()
        wrist_train = wrist_train.as_matrix()
#        X_train.append(features(fs=fs,batch_size=batch_size,arm=arm_train, wrist=wrist_train))
        X = features(fs=fs,batch_size=batch_size,arm=arm_train, wrist=wrist_train)
        X = sc.transform(X)
        df = pd.DataFrame(X,columns=range(0,84))
        df.to_csv('Test_Features/'+i+'/features.csv', index=False, header=False)     