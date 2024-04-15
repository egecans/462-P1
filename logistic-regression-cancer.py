from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = None 

    def initialize_weights(self, num_features):
        self.weights = np.zeros(num_features)
        self.bias = 0
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_pred, y_true):        
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 =  (1 - y_true) * np.log(1 - y_pred + epsilon)
        loss = -np.mean(y1 + y2)
        return loss

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.initialize_weights(n_features)

        for i in range(self.num_iter):
            z = X.dot(self.weights) + self.bias
            A = self.sigmoid(z)
          
            loss = self.compute_loss(A, y)

            # calculate gradients         
            tmp = ( A - y.T )         
            tmp = np.reshape( tmp, n_samples )         
            dW = np.dot( X.T, tmp ) / n_samples          
            db = np.sum( tmp ) / n_samples  
            
            # update weights     
            self.weights = self.weights - self.lr * dW     
            self.bias = self.bias - self.lr * db 

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss}")

    def predict(self, x):
        tmp = x.dot(self.weights) + self.bias
        Z = self.sigmoid(tmp)    
        Y = np.where( Z > 0.5, 1, 0 ) 
        return Y 

def dataset_split(df):
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    split_index = int(0.8 * len(df))

    train_df = df[:split_index]
    val_df = df[split_index:]

    X_train = train_df.drop(columns=["Diagnosis", "ID"])
    y_train = train_df["Diagnosis"]

    X_val = val_df.drop(columns=["Diagnosis", "ID"])
    y_val = val_df["Diagnosis"]

    return X_train, target_fix(y_train), X_val, target_fix(y_val) 

def target_fix(target):
    new_target = []
    target = target.to_numpy()
    for idx in range(len(target)):
        if target[idx] == "M":
            new_target.append(0)
        else:
            new_target.append(1) 
    return np.array(new_target)

def accuracy_score(pred, y):
    tp = 0
    for i in range(len(pred)):
        if pred[i] == y[i]:
            tp += 1

    return tp / len(pred)

# fetch breast cancer dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

data = breast_cancer_wisconsin_diagnostic.data.original

X_train, y_train, X_val, y_val = dataset_split(data)

model = LogisticRegression(lr=0.01, num_iter=1000)
model.fit(X_train, y_train)
pred = model.predict(X_val)
val_acc = accuracy_score(pred, y_val)


print("Validation Accuracy:" + str(val_acc))