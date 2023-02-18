import numpy as np
import pandas as pd
#import pygame as pg
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')

#X = data.iloc[:, :2]
X = data.iloc[:, :2].values
Y = data.iloc[:,4]

Y = np.where(Y == 'Setosa', 0, 1)



#Split the data into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


X__train = np.c_[np.ones((len(X_train),1)),X_train]
X__test = np.c_[np.ones((len(X_test),1)),X_test]
#print(X__train)
#theta = np.random.random(X__train.shape[1])








plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='Virginica')

plt.xlabel('sepal length [cm]')
plt.ylabel('sepal width [cm]')
plt.legend(loc='upper left')

plt.show()



def sigmoid(X, theta):
    
    z = np.dot(X, theta[1:]) + theta[0]
    
    return 1.0 / ( 1.0 + np.exp(-z))

def lrCostFunction(y, hx):
  
    # compute cost for given theta parameters
    j = -y.dot(np.log(hx)) - ((1 - y).dot(np.log(1-hx)))
    
    return j

def lrGradient(X, y, theta, alpha, num_iter):
    # empty list to store the value of the cost function over number of iterations
    cost = []
    
    for i in range(num_iter):
        # call sigmoid function 
        hx = sigmoid(X, theta)
        # calculate error
        error = hx - y
        # calculate gradient
        grad = X.T.dot(error)
        # update values in theta
        theta[0] = theta[0] - alpha * error.sum()
        theta[1:] = theta[1:] - alpha * grad
        
        cost.append(lrCostFunction(y, hx))
        
    return cost





def lrPredict(X):
    
    return np.where(sigmoid(X,theta) >= 0.5, 1, 0)


# m = Number of training examples
# n = number of features
m, n = X__train.shape

# initialize theta(weights) parameters to zeros
theta = np.zeros(1+n)

# set learning rate to 0.01 and number of iterations to 500
alpha = 0.001
num_iter = 50000

cost = lrGradient(X__train, Y_train, theta, alpha, num_iter)









# Make a plot with number of iterations on the x-axis and the cost function on y-axis
plt.plot(range(1, len(cost) + 1), cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Logistic Regression')
plt.show()

print ('\n Logisitc Regression bias(intercept) term :', theta[0])
print ('\n Logisitc Regression estimated coefficients :', theta[1:])



hx = sigmoid(X__test, theta)
m = len(Y_test)
#error
for i in range(m):
    error = lrCostFunction(Y_test,hx)
print(error)


#from matplotlib.colors import ListedColormap

#def plot_decision_boundry(X, y, classifier, h=0.02):
#    # h = step size in the mesh
  
#    # setup marker generator and color map
#    markers = ('s', 'x', 'o', '^', 'v')
#    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#    cmap = ListedColormap(colors[:len(np.unique(y))])

#    # plot the decision surface
#    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
#                         np.arange(x2_min, x2_max, h))
#    Z = classifier(np.array([xx1.ravel(), xx2.ravel()]).T)
#    Z = Z.reshape(xx1.shape)
#    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#    plt.xlim(xx1.min(), xx1.max())
#    plt.ylim(xx2.min(), xx2.max())

#    # plot class samples
#    for idx, cl in enumerate(np.unique(Y)):
#        plt.scatter(x=X[Y == cl, 0], Y=X[Y == cl, 1],
#                    alpha=0.8, c=cmap(idx),
#                    marker=markers[idx], label=cl)
        



#plot_decision_boundry(X__train,Y, classifier=lrPredict(X__test))
#plt.title('Logistic Regression - Gradient Descent')
#plt.xlabel('sepal length ')
#plt.ylabel('sepal width ')
#plt.legend(loc='upper left')
#plt.tight_layout()

