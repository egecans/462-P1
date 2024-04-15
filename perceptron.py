import numpy as np
import matplotlib.pyplot as plt
import sys

# Assuming the first column of your data is the label
features = np.load('data_small.npy')
labels = np.load('label_small.npy')

features_large = np.load('data_large.npy')
labels_large = np.load('label_large.npy')

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
def perceptron(features, labels, learning_rate, epochs, weights):
    
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
            print("Iteration Count: " + str(len(misclassification_steps)))
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


# shuffling data to observe iteration count changes
def shuffle_data_order(features, labels):
    # Generate a sequence of indices and shuffle them to get different initial points
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)

    # Use the shuffled indices to reorder both the features and labels
    shuffled_features = features[indices]
    shuffled_labels = labels[indices]
    return shuffled_features, shuffled_labels

def repeat_training(features, labels, learning_rate, epochs, num_training):

    step_counts = []
    weights = []

    # to initialize different datapoints we need to random weights
    initial_weights_sets = []
    for _ in range(num_training):
        initial_weights_sets.append(np.random.randn(features.shape[1]))
    
    for i in range(num_training):

        # Generate a sequence of indices and shuffle them to get different initial points
        indices = np.arange(features.shape[0])
        np.random.shuffle(indices)

        # Use the shuffled indices to reorder both the features and labels
        shuffled_features = features[indices]
        shuffled_labels = labels[indices]

        trained_weights, features_with_label, misclassifications = perceptron(shuffled_features, shuffled_labels, learning_rate, epochs, initial_weights_sets[i])
        
        # add step counts to plot them
        step_counts.append(len(misclassifications))
        weights.append(trained_weights)

        print('Trained weights:', trained_weights)
        plot_funct(features_with_label, trained_weights)
        plot_misclassifications(misclassifications, learning_rate)
    
    return step_counts, weights


# Train the Perceptron
def train_once(features, labels):
    learning_rate = 10
    epochs = 1000
    # Initialize weights to zero or a small random value
    # w0, w1, w2
    weights = np.zeros(features.shape[1])
    trained_weights, features_with_label, misclassification_in_each_step = perceptron(features, labels, learning_rate, epochs, weights)

    print('Trained weights:', trained_weights)
    plot_funct(features_with_label, trained_weights)
    plot_misclassifications(misclassification_in_each_step, learning_rate)


# plotting misclassification w.r.t learning rate
def plot_iterations(step_counts):
    plt.plot(step_counts)
    plt.xlabel('step')
    plt.ylabel('epoch')
    plt.show() 


learning_rate = 0.1
num_of_iterations = 3
max_epochs = 1000

print("Which option would you prefer \n 1 for train small dataset once \n 2 for train large dataset once \n 3 for train small dataset 3 times with different initializations \n 4 for train large dataset 3 times with different initializations")
option = input()

# Execute functions based on the input
if option == "1":
    train_once(features, labels)
elif option == "2":
    train_once(features_large, labels_large)
elif option == "3":
    step_counts, trained_weights = repeat_training(features, labels, learning_rate, max_epochs, num_of_iterations)
    #print(f"Step counts: {step_counts}, Trained weights: {trained_weights}")
elif option == "4":
    step_counts, trained_weights = repeat_training(features_large, labels_large, learning_rate, max_epochs, num_of_iterations)
    #print(f"Step counts: {step_counts}, Trained weights: {trained_weights}")
else:
    print("Invalid option! " + str(option) + " Choose from 1, 2, 3, or 4.")