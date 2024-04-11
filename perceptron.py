import numpy as np
import matplotlib.pyplot as plt

# Assuming the first column of your data is the label
features = np.load('data_large.npy')
labels = np.load('label_large.npy')

# it returns 1 if dot product >= 0 else return -1
def prediction_result(dot_product):
    if dot_product >= 0:
        return 1
    else:
        return -1

# it returns false if output and label are the same, else return true
def predicted_false(output, label):
    return(prediction_result(output) != label)


# Perceptron Learning Algorithm
def perceptron(features, labels, learning_rate, epochs):
    
    # Initialize weights to zero or a small random value
    # w0, w1, w2
    weights = np.zeros(features.shape[1])
    misclassification_steps = []


    for epoch in range(epochs):
        # Initialize misclassification count for this epoch
        misclassified = 0
        
        # predicted labels to return at the end, in each epoch it's empty
        predicted_label = []

        for i in range(len(features)):
            
            # Calculate the dot product 
            dot_product = np.dot(features[i], weights)

            predicted_label.append(prediction_result(dot_product))

            # Check for misclassification
            if predicted_false(dot_product, labels[i]):
                # Update weights for misclassifications
                for j in range(len(weights)):
                    weights[j] += learning_rate * labels[i] * features[i][j]
                misclassified += 1
        
        # Optionally, print the number of misclassified points in each epoch
        misclassification_steps.append(misclassified)

        # If no misclassifications, the model is perfect under current data, stop the algorithm
        if misclassified == 0:
            break
    
    # change list to np.array
    np_predicted_label = np.array(predicted_label)
    # make [1,-1,1] = [[1],[-1],[1]] to handle dimensions
    np_predicted_label = np_predicted_label.reshape(len(predicted_label), 1)
    # concatenate predicted labels at the end of feature vector 
    features_with_label = np.hstack((features,np_predicted_label))

    return weights, features_with_label, misclassification_steps


# Plotting
def plot_funct(feature_label, weights):

    # Extracting features
    x1 = feature_label[:, 1]
    x2 = feature_label[:, 2]
    labels = feature_label[:, 3]
    
    # Plotting the data points
    # x1[labels == 1] is condition property of numpy, it gets x1 points that has label 1
    plt.scatter(x1[labels == 1], x2[labels == 1], color='blue', label='Class 1')
    plt.scatter(x1[labels == -1], x2[labels == -1], color='red', label='Class -1')
    
    # Calculating line parameters
    # continuous x values on the line
    x_values = np.array(plt.gca().get_xlim())
    # slope of y
    y_values = -(weights[1] * x_values + weights[0]) / weights[2]
    
    # Plotting the decision boundary
    plt.plot(x_values, y_values, label='Decision Boundary')
    plt.xlabel('x2')
    plt.ylabel('x1')
    plt.legend()
    plt.show()


# plotting misclassification w.r.t learning rate
def plot_misclassifications(misclassification_in_each_step, learning_rate):
    plt.plot(misclassification_in_each_step)
    plt.xlabel('iterations')
    plt.ylabel('misclassified')
    plt.title('Learning rate: ' + str(learning_rate))
    plt.show() 


# Train the Perceptron
learning_rate = 10
epochs = 1000
trained_weights, features_with_label, misclassification_in_each_step = perceptron(features, labels, learning_rate, epochs)

print('Trained weights:', trained_weights)
plot_funct(features_with_label, trained_weights)
plot_misclassifications(misclassification_in_each_step, learning_rate)