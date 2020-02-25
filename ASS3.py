import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

data = pd.read_csv("house_data_complete.csv")
dataFold = data.copy()
data.dropna(axis =1, inplace=True )
y=data['price']
data.drop(["id","date","price","grade","sqft_lot15",'sqft_living15'],axis=1, inplace=True)
dataFold.drop(["id","date","grade","sqft_lot15",'sqft_living15'],axis=1, inplace=True)
dataFold.dropna(axis =1, inplace=True )

Xtrain, Xtest,Ytrain, Ytest = train_test_split(data ,y, test_size=0.2)
#print(Xtrain)
Xtrain, Xval,Ytrain, Yval = train_test_split(Xtrain,Ytrain,test_size=0.2)

m = Ytrain.size # 3add el rows (examples)


def  featureNormalize(X):

    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X-mu)/sigma
    return X_norm


def computeCostMulti(X, y, theta):
    J = np.dot((np.dot(X, theta) - y), (np.dot(X, theta) - y)) / (2 * m)
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    J_history = []

    for i in range(num_iters):

        theta = theta - (alpha / m) * (np.dot(X, theta.T) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history

alpha = 0.3
num_iters = 50
# --------------------------------------------------------
X_norm1 = featureNormalize(Xtrain)
ones = np.ones(Xtrain.shape[0])
X_norm1["x0"] = ones
theta1 = np.zeros(np.size(X_norm1, 1))
theta1, J_history = gradientDescentMulti(X_norm1, Ytrain, theta1, alpha, num_iters)

plt.figure(2)
plt.plot(np.arange(len(J_history)), J_history,lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
#plt.show()
# ---------------------------- Hypothesis 2 train

X_norm2 = featureNormalize(Xtrain)
X_norm2["condition"] = np.square(X_norm2["condition"])
ones = np.ones(Xtrain.shape[0])
X_norm2["x0"] = ones
theta2 = np.zeros(np.size(X_norm2, 1))
theta2, J_history = gradientDescentMulti(X_norm2, Ytrain, theta2, alpha, num_iters)

plt.figure(3)
plt.plot(np.arange(len(J_history)), J_history,lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()
# ------------------------------------ Hypothesis 3

X_norm3 = featureNormalize(Xtrain)
X_norm3["bedrooms"] = np.square(X_norm3["bedrooms"])
ones = np.ones(Xtrain.shape[0])
X_norm3["x0"] = ones
theta3 = np.zeros(np.size(X_norm3, 1))
theta3, J_history = gradientDescentMulti(X_norm2, Ytrain, theta3, alpha, num_iters)

plt.figure(4)
plt.plot(np.arange(len(J_history)), J_history,lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
#plt.show()

# ------------------------------------- validation1
X_norm4 = featureNormalize(Xval)
ones = np.ones(Xval.shape[0])
X_norm4["x0"] = ones
C = computeCostMulti(X_norm4, Yval, theta1)
print("val1",C)

X_norm7 = featureNormalize(Xtest)
ones = np.ones(Xtest.shape[0])
X_norm7["x0"] = ones
Ct= computeCostMulti(X_norm7, Ytest, theta1)
print("ct",Ct)
# --------------------------- validation2
X_norm5 = featureNormalize(Xval)
X_norm5["condition"] = np.square(X_norm5["condition"])
ones = np.ones(Xval.shape[0])
X_norm5["x0"] = ones
C2= computeCostMulti(X_norm5, Yval, theta2)
print("val2",C2)

X_norm8= featureNormalize(Xtest)
X_norm8["condition"] = np.square(X_norm8["condition"])
ones = np.ones(Xtest.shape[0])
X_norm8["x0"] = ones
Ct2= computeCostMulti(X_norm8, Ytest, theta2)
print("ct2",Ct2)
#---------------- validation3
X_norm6 = featureNormalize(Xval)
X_norm6["bedrooms"] = np.square(X_norm6["bedrooms"])
ones = np.ones(Xval.shape[0])
X_norm6["x0"] = ones
C3= computeCostMulti(X_norm6, Yval, theta3)
print("val2",C3)

X_norm9 = featureNormalize(Xtest)
X_norm9["bedrooms"] = np.square(X_norm9["bedrooms"])
ones = np.ones(Xtest.shape[0])
X_norm9["x0"] = ones
Ct3= computeCostMulti(X_norm9, Ytest, theta3)
print("ct3",Ct3)
# ---------------------- Kfold

kf=KFold(n_splits=3,shuffle=True,random_state=2)
result= next(kf.split(dataFold),None)

trainK= dataFold.iloc[result[0]]
trainky= trainK.price
trainK = trainK.drop('price',axis=1)

testK = dataFold.iloc[result[1]]
testKy = testK.price
testK = testK.drop('price', axis=1)

print(trainK)
print(testK)

