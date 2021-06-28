import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from numpy import save
from numpy import load

def preprocess(data):
    return preprocessing.scale(data)

def closed_form(X,y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

def RBF_design_matrix(X,C):
    N = X.shape[0]
    kmeans = KMeans(n_clusters=C, random_state=123).fit(X)
    sig = np.std(X)
    U = np.zeros((N,C))

    for n in range(N):
        for c in range(C):
            U[n][c] = np.exp(-np.linalg.norm(X[n] - kmeans.cluster_centers_[c])**2/sig**2)
    return U
    
def RBF_pseudoinverse(X, y, C):
    U = RBF_design_matrix(X,C)
    w_pseudoinv = np.linalg.inv(U.T @ U) @ U.T @ y
    return w_pseudoinv

def rbf(data,center,sigma):
    return np.exp(-(np.linalg.norm(data-center)**2) / (sigma**2))

def RBF_SGD(X, y, C, lr=0.1, epochs=300):
    N = X.shape[0]
    kmeans = KMeans(n_clusters=C, random_state=123).fit(X)
    sig = np.std(X)
    w_RBF = np.random.randn(C,1)
    U = np.zeros((N,C))

    for n in range(N):
        for i in range(C):
            U[n][i] = np.exp(-np.linalg.norm(X[n] - kmeans.cluster_centers_[i])**2/sig**2)

    losses = []
    for ep in range(epochs):
        for n in range(N):
            RBF = np.array([rbf(X[n],c,sig) for c in kmeans.cluster_centers_])
            RBF = np.expand_dims(RBF,1)

            pred = RBF.T @ w_RBF

            loss = -(y[n] - pred)

            w_RBF = w_RBF - lr*RBF*loss

        predict = U @ w_RBF
        error = np.sum(np.absolute(y - predict))
        losses.append(error)

        if ep%1 == 0:
            print('Epoch:',ep,'Loss:',error)
            
    return w_RBF, losses

def train(q, C, lr=0.1, epochs=50):
    #from numpy import save
    #save('q_table_200episode_40state_2.npy', q_table)
    load_q_table = load('q_table_200episode_40state.npy')
    #load_q_table = q_table

    X = []
    y = []
    for i in range(len(load_q_table)):
        for j in range(len(load_q_table[i])):
            for k in range(len(load_q_table[i][j])):
                if load_q_table[i][j][k] != 0:
                    X.append([i,j,k])
                    y.append([load_q_table[i][j][k]])

    X = np.array(X)
    y = np.array(y)

    scaler1 = StandardScaler(with_mean=False).fit(X)
    scaler2 = StandardScaler().fit(y)

    X = scaler1.transform(X)
    y = scaler2.transform(y)
    
    N = X.shape[0]
    #C = 100

    U = np.zeros((N,C))
    N = X.shape[0]
    kmeans = KMeans(n_clusters=C, random_state=123).fit(X)
    sig = np.std(X)

    for n in range(N):
        for c in range(C):
            U[n][c] = rbf(X[n],kmeans.cluster_centers_[c],sig)

    w_RBF = np.random.randn(C,1)

    errors = []
    for ep in range(epochs):
        for n in range(N):
            RBF = np.array([rbf(X[n], c, sig) for c in kmeans.cluster_centers_])
            RBF = np.expand_dims(RBF,1)

            pred = w_RBF.T @ RBF

            loss = y[n] - pred
            #print(y[n], pred)

            derivative = loss*(-RBF)

            w_RBF = w_RBF - lr*derivative

        predict = U @ w_RBF
        error = np.sum(np.abs(y - predict))/N
        errors.append(error)
        print('Epoch:', ep, 'Loss:', error)
        
    
    return U, w_RBF, kmeans, sig, scaler1, scaler2, errors
