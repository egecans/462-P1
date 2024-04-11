import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    return np.load(file_name)

def plot_dataset(X, Y, weights, title):
    # Plot the data points
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Set1, edgecolor='k')
    
    # Create a grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Plot the decision boundary
    Z = np.dot(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()], weights)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, levels=[-1,0,1], colors=['blue','red'])
    
    # Titles and labels
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

def perceptron(X, Y, learning_rate, epochs):
    X_with_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    weights = np.zeros(X_with_bias.shape[1])
    iteration = 0

    for epoch in range(epochs):
        no_errors = True
        for i in range(len(X)):
            linear_output = np.dot(X_with_bias[i], weights)
            prediction = 1 if linear_output > 0 else 0
            if prediction != Y[i]:
                no_errors = False
                weights += learning_rate * (Y[i] - prediction) * X_with_bias[i]
        if no_errors:
            iteration = epoch + 1
            break
    
    return weights, iteration

def experiment(datasets):
    for dataset_name in datasets:
        data = load_data(f'{dataset_name}.npy')
        X = data[:, 1:]  
        Y = data[:, 0]   

        # Train the Perceptron
        learning_rate = 0.1
        epochs = 1000

        final_weights, iterations_to_converge = perceptron(X, Y, learning_rate, epochs)
        print(f'{dataset_name}: Converged in {iterations_to_converge} iterations')

        plot_dataset(X, Y, final_weights, f'Decision Boundary for {dataset_name}')

def sensitivity_to_initialization(dataset_name, num_experiments):
    data = load_data(f'{dataset_name}.npy')
    X = data[:, 1:]  
    Y = data[:, 0]   

    # Record weights for each experiment
    weights_all_experiments = []

    for _ in range(num_experiments):
        initial_weights = np.random.rand(X.shape[1] + 1)  # Random initialization
        final_weights, _ = perceptron(X, Y, 0.1, 1000)
        weights_all_experiments.append(final_weights)
    
    return weights_all_experiments

# Run the experiments
experiment(['data_small', 'data_large'])

# Analyze sensitivity to initial weights
num_experiments = 10
weights_small_dataset = sensitivity_to_initialization('data_small', num_experiments)

# Plotting the weights to compare
plt.figure()
for i, weights in enumerate(weights_small_dataset):
    plt.plot(weights, label=f'Experiment {i+1}')
plt.title('Weights comparison over different initializations')
plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.legend()
plt.show()

# After the plots are displayed, you can add your explanation regarding the sensitivity to initialization.
