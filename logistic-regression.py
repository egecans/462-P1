from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000, reg_param=0):
        self.lr = lr
        self.num_iter = num_iter
        self.reg_param = reg_param
        self.weights = None
        self.bias = None 

    def initialize_weights(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_pred, y_true):
        size = len(y_true)
        
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 =  (1 - y_true) * np.log(1 - y_pred + epsilon)
        loss = -np.mean(y1 + y2)

        reg_term = (self.reg_param / (2*size)) * np.sum(self.weights ** 2)
        return loss + reg_term

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.initialize_weights(n_features)

        loss_iter = []

        for i in range(self.num_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
                
            # Compute loss
            loss = self.compute_loss(y_pred, y)

            tmp = (y_pred - y.T )         
            tmp = np.reshape( tmp, n_samples )         
            dW = np.dot( X.T, tmp ) + 2 * self.reg_param * self.weights  / n_samples         
            db = np.sum( tmp ) / n_samples  
            
            # update weights     
            self.weights = self.weights - self.lr * dW     
            self.bias = self.bias - self.lr * db 

            loss_iter.append(loss)

        return loss_iter

    def predict(self, x):
        tmp = x.dot(self.weights) + self.bias
        Z = self.sigmoid(tmp)    
        Y = np.where( Z > 0.5, 1, 0 ) 
        return Y 

class LogisticRegressionSGD(LogisticRegression):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        
        loss_epoch = []

        # Stochastic Gradient Descent
        for i in range(self.num_iter):
            # Shuffle the data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Iterate over each training example
            for j in range(n_samples):
                xi = X_shuffled[j, :].reshape(1, -1)
                yi = y_shuffled[j]
                
                # Forward pass
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self.sigmoid(z)
                
                # Compute loss
                loss = self.compute_loss(y_pred, yi)

                # calculate gradients         
                dW = np.dot(xi.T, (y_pred - yi))
                reg_gradient = 2 * self.reg_param * self.weights
                dW = (dW + reg_gradient)/ n_samples 
                db = (y_pred - yi) / n_samples 
                self.weights -= self.lr * dW
                self.bias -= self.lr * db

            loss_epoch.append(loss)

        return loss_epoch

# to perform 5-fold cross-validation
def k_fold_cross_val(X, y, random_seed=None):
    num_folds = 5
    fold_size = len(X) // num_folds

    params = [0.001, 0.01, 0.1, 1, 10] # ???

    X_random, y_random = dataset_shuffle(X, y, random_seed)

    avg_acc_param = []

    for param in params:
        fold_acc = []

        model = LogisticRegression(lr=0.1, num_iter=1000, reg_param=param)

        for fold in range(num_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size

            X_val = X_random[val_start:val_end]
            y_val = y_random[val_start:val_end]

            X_train = np.concatenate([X_random[:val_start], X_random[val_end:]])
            y_train = np.concatenate([y_random[:val_start], y_random[val_end:]])

            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            val_acc = accuracy_score(pred, y_val)
            
            print("Validation per fold: " + str(val_acc))
            fold_acc.append(val_acc)

        avg_acc = np.mean(fold_acc)
        print("Validation per param: " + str(avg_acc))

        avg_acc_param.append(avg_acc)

    best_param = np.argmax(avg_acc_param)
    return params[best_param]

def accuracy_score(pred, y):
    tp = 0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            tp += 1

    return tp / len(pred)

def normalization(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    normalized_data = (X - min_vals) / (max_vals - min_vals)
    return normalized_data

def dataset_shuffle(X, y, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.random.permutation(len(X)) # this might change
    X_random = X[indices]
    y_random = y[indices]

    return X_random, y_random

def dataset_split(X, y):
    split_index = int(0.8 * len(X))

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_val = X[split_index:]
    y_val = y[split_index:]

    return X_train, y_train, X_val, y_val

# fetching the dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 

X = normalization(X.values)

y_binary = []
for value in y.values:
    if value == "Cammeo":
        y_binary.append(0)
    else: 
        y_binary.append(1)
y_binary = np.array(y_binary).reshape(len(y),1)

# 5-fold cross-validation to choose best regularization parameter
#k_fold_cross_val(X, y_binary) 

X_random, y_random = dataset_shuffle(X, y_binary)
X_train, y_train, X_val, y_val = dataset_split(X_random, y_random)

'''
# Logistic Regression trained with full batch (GD)
model_GD = LogisticRegression(lr=0.01, num_iter=1000, reg_param=0)
loss_iter = model_GD.fit(X_train, y_train)
pred_gd_train = model_GD.predict(X_train)
val_acc_gd_train = accuracy_score(pred_gd_train, y_train)
pred_gd = model_GD.predict(X_val)
val_acc_gd = accuracy_score(pred_gd, y_val)
print("Training Accuracy of Logistic Regression with GD:" + str(val_acc_gd_train))
print("Validation Accuracy of Logistic Regression with GD:" + str(val_acc_gd))
'''
# Logisctic Regression trained with SGD
model_SGD = LogisticRegressionSGD(lr=0.01, num_iter=1000, reg_param=0)
loss_epoch_2 = model_SGD.fit(X_train, y_train)
pred_sgd_train = model_SGD.predict(X_train)
val_acc_sgd_train = accuracy_score(pred_sgd_train, y_train)
pred_sgd = model_SGD.predict(X_val)
val_acc_sgd = accuracy_score(pred_sgd, y_val)
print("Training Accuracy of Logistic Regression with SGD with LR = 0.01:" + str(val_acc_sgd_train))
print("Validation Accuracy of Logistic Regression with SGD:" + str(val_acc_sgd))

model_SGD = LogisticRegressionSGD(lr=0.001, num_iter=1000, reg_param=0)
loss_epoch_3 = model_SGD.fit(X_train, y_train)
pred_sgd_train = model_SGD.predict(X_train)
val_acc_sgd_train = accuracy_score(pred_sgd_train, y_train)
pred_sgd = model_SGD.predict(X_val)
val_acc_sgd = accuracy_score(pred_sgd, y_val)
print("Training Accuracy of Logistic Regression with SGD with LR = 0.001:" + str(val_acc_sgd_train))
print("Validation Accuracy of Logistic Regression with SGD:" + str(val_acc_sgd))

model_SGD = LogisticRegressionSGD(lr=0.1, num_iter=1000, reg_param=0)
loss_epoch_1 = model_SGD.fit(X_train, y_train)
pred_sgd_train = model_SGD.predict(X_train)
val_acc_sgd_train = accuracy_score(pred_sgd_train, y_train)
pred_sgd = model_SGD.predict(X_val)
val_acc_sgd = accuracy_score(pred_sgd, y_val)
print("Training Accuracy of Logistic Regression with SGD with LR = 0.1:" + str(val_acc_sgd_train))
print("Validation Accuracy of Logistic Regression with SGD:" + str(val_acc_sgd))

'''
# Logisctic Regression regularized by L2 norm trained with GD
model_GD_r = LogisticRegression(lr=0.01, num_iter=500, reg_param=0.01)
model_GD_r.fit(X_train, y_train)
pred_gdr_train = model_GD_r.predict(X_train)
val_acc_gdr_train = accuracy_score(pred_gdr_train, y_train)
pred_gdr = model_GD_r.predict(X_val)
val_acc_gdr = accuracy_score(pred_gdr, y_val)
print("Training Accuracy of Logistic Regression regularized by L2 norm with GD:" + str(val_acc_gdr_train))
print("Validation Accuracy of Logistic Regression regularized by L2 norm with GD:" + str(val_acc_gdr))

# Logisctic Regression regularized by L2 norm trained with SGD
model_SGD_r = LogisticRegressionSGD(lr=0.01, num_iter=500, reg_param=0.01)
model_SGD_r.fit(X_train, y_train)
pred_sgdr_train = model_SGD_r.predict(X_train)
val_acc_sgdr_train = accuracy_score(pred_sgdr_train, y_train)
pred_sgdr = model_SGD_r.predict(X_val)
val_acc_sgdr = accuracy_score(pred_sgdr, y_val)
print("Training Accuracy of Logistic Regression regularized by L2 norm with SGD:" + str(val_acc_sgdr_train))
print("Validation Accuracy of Logistic Regression regularized by L2 norm with SGD:" + str(val_acc_sgdr))
'''

#To plot losses between GD and SGD
'''
y1 = np.array(loss_iter)
y2 = np.array(loss_epoch_2)
plt.plot(y1)
plt.plot(y2)

plt.legend(["GD", "SGD"])

plt.savefig("losses.png",format="png")

plt.show()
'''

#To plot losses SGD with different learning rates

y1 = np.array(loss_epoch_1)
y2 = np.array(loss_epoch_2)
y3 = np.array(loss_epoch_3)
plt.plot(y1)
plt.plot(y2)
plt.plot(y3)

plt.legend(["lr = 0.1", "lr = 0.01", "lr = 0.001"])

plt.savefig("losses.png",format="png")

plt.show()