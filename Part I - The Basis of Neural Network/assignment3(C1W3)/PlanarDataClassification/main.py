# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from functools import reduce
from model_function import *
import operator


# GRADED FUNCTION: nn_model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False, lr=1.12):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    ### START CODE HERE ### (≈ 5 lines of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
#     print("Before Update parameters: ")
#     print('W1', W1)
#     print('b1', b1)
#     print('W2', W2)
#     print('b2', b2)
    ### END CODE HERE ###
    
    # Loop (gradient descent)

    for i in range(0,num_iterations):
         
        ### START CODE HERE ### (≈ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
       #print("grads:", grads)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=lr)
        #print("new parameters:", parameters)
        
        ### END CODE HERE ###
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

# GRADED FUNCTION: predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    ### END CODE HERE ###
    
    return predictions


if __name__ == "__main__":

    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}

    ### START CODE HERE ### (choose your dataset)
    while(True):
        print()
        print('Please select the dataset as follows: ')
        print('1 - > noisy_circles ')
        print('2 - > noisy_moons')
        print('3 - > blobs')
        print('4 - > gaussian_quantiles')
        print('5 -> flower data')
        print()
        data = input('Please input your answer(1, 2, 3,4, 5). q is quit : ')
        if (data == 'q'):
            break
        if (data == '1'):
            X_new, Y_new = datasets["noisy_circles"]
            X_new, Y_new = X_new.T, Y_new.reshape(1, Y_new.shape[0])
            print("noisy_circles")
        if (data == '2'):
            X_new, Y_new = datasets["noisy_moons"]
            X_new, Y_new = X_new.T, Y_new.reshape(1, Y_new.shape[0])
            print("noisy_moons")
        if (data == '3'):
            X_new, Y_new = datasets["blobs"]
            X_new, Y_new = X_new.T, Y_new.reshape(1, Y_new.shape[0])
            print("blobs")
        if (data == '4'):
            X_new, Y_new = datasets["gaussian_quantiles"]
            X_new, Y_new = X_new.T, Y_new.reshape(1, Y_new.shape[0])
            print("gaussian_quantiles")
        if (data == '5'):
            X_new, Y_new = load_planar_dataset()
            print("flower data")

        
        ### END CODE HERE ###

        
        #X_new, Y_new = X_new.T, Y_new.reshape(1, Y_new.shape[0])

        # make blobs binary
        if data == "3":
            Y_new = Y_new % 2

        # Visualize the data
        plt.scatter(X_new[0, :], X_new[1, :], c=Y_new.flatten(), s=40, cmap=plt.cm.Spectral);
        plt.show()

        # Build a model with a n_h-dimensional hidden layer
        parameters = nn_model(X_new, Y_new, n_h = 4, num_iterations = 10000, print_cost=False)
        #print('parameters', parameters)
        # Plot the decision boundary
        plot_decision_boundary(lambda x: predict(parameters, x.T), X_new, Y_new.reshape(-1,))
        plt.title("Decision Boundary for hidden layer size " + str(4))
        plt.show()

        # Print accuracy
        predictions = predict(parameters, X_new)
        print ('Accuracy: %d' % float((np.dot(Y_new,predictions.T) + np.dot(1-Y_new,1-predictions.T))/float(Y_new.size)*100) + '%')